"""FastAPI service: runs the dealer simulation and serves the demo page."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .sim import BPS, Params, run, sweep

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Dealer hedge-vs-warehouse", docs_url="/api/docs")


class Req(BaseModel):
    p_win: float = 0.35
    skill_edge: float = 0.10
    skill_disp: float = 0.04
    vol_regime: str = "common"
    vol_disp: float = 0.5
    price_snr: float = 0.35
    hedge_cost_bps: float = 0.5
    n_clients: int = 60
    n_days: int = 500
    tickets_per_day: int = 12
    train_frac: float = 0.40
    seed: int = 7
    with_sweep: bool = Field(default=True)


def _thin(a: np.ndarray, n: int = 240) -> list[float]:
    if len(a) <= n:
        return [round(float(v), 2) for v in a]
    idx = np.linspace(0, len(a) - 1, n).astype(int)
    return [round(float(v), 2) for v in a[idx]]


def _hist(v: np.ndarray, lo: float, hi: float, bins: int) -> list[float]:
    h, _ = np.histogram(v, bins=bins, range=(lo, hi))
    return [int(c) for c in h]


TAPE_MAX = 1500


def _tape(p, tk: dict, dec: dict, live: np.ndarray) -> dict[str, Any]:
    """Per-ticket blotter over the live window: what the two models said
    before the ticket resolved, what actually happened, and what the desk
    made or lost against the always-hedge benchmark on that ticket alone."""
    s = tk["side"][live]
    notion = tk["notional"][live] * 1e6
    r = tk["r"][live]
    wh = dec["warehouse"][live]

    pnl = np.where(wh, -r * notion, -p.hedge_cost_bps * BPS * notion)
    base = np.full(len(r), -p.hedge_cost_bps * BPS) * notion
    cum_gain = np.cumsum(pnl - base)          # cumulative over EVERY live ticket

    post_up = dec["post_up"][live]
    p_right = np.where(s > 0, post_up, 1 - post_up)   # combined P(client on the right side)

    n = len(r)
    idx = (np.arange(n) if n <= TAPE_MAX
           else np.unique(np.linspace(0, n - 1, TAPE_MAX).astype(int)))

    # keep this compact -- the whole payload is refetched on every slider move.
    # r_bps = side * x_bps and base = pnl - gain, so neither is sent.
    rd = lambda a, d: [round(float(v), d) for v in a[idx]]
    ri = lambda a: [int(round(float(v))) for v in a[idx]]
    return {
        "n": int(len(idx)), "total": int(n), "sampled": bool(n > TAPE_MAX),
        "day": [int(v) for v in tk["day"][live][idx]],
        "cli": [int(v) for v in tk["cli"][live][idx]],
        "ins": [int(v) for v in tk["ins"][live][idx]],
        "side": [int(v) for v in s[idx]],
        "notional": rd(notion / 1e6, 2),
        "p_up": rd(dec["p_up"][live], 3),                 # model 1: P(underlying up)
        "p_cli": rd(0.5 + dec["q_used"][live], 3),        # model 2: P(this client wins)
        "p_right": rd(p_right, 3),                        # combined, drives the decision
        "warehouse": [int(v) for v in wh[idx]],
        "x_bps": rd(tk["x"][live] / BPS, 1),
        "pnl": ri(pnl), "gain": ri(pnl - base), "cum_gain": ri(cum_gain),
    }


def _payload(res: dict[str, Any], with_sweep: bool) -> dict[str, Any]:
    p, tk, cm, pm, dec = res["p"], res["tk"], res["cm"], res["pm"], res["dec"]
    live = res["live"]

    # ---- equity curves ---------------------------------------------------
    curves = {k: _thin(np.cumsum(b["daily"]))
              for k, b in res["books"].items()}

    # ---- client population ------------------------------------------------
    q, isw = tk["q_true"], tk["is_win"]
    LO, HI, NB = -0.30, 0.30, 30
    pop = {
        "lo": LO, "hi": HI, "bins": NB,
        "winners": _hist(q[isw], LO, HI, NB),
        "losers": _hist(q[~isw], LO, HI, NB),
        "n_win": int(isw.sum()), "n_lose": int((~isw).sum()),
    }

    # ---- client model: estimated vs true ---------------------------------
    n_tr = cm["n_trades_final"]
    scatter = [{"t": round(float(a), 4), "e": round(float(b), 4), "n": int(c)}
               for a, b, c in zip(tk["q_true"], cm["q_final"], n_tr)]
    ok = n_tr > 5
    corr = float(np.corrcoef(tk["q_true"][ok], cm["q_final"][ok])[0, 1]) if ok.sum() > 3 else None

    # ---- decision surface: warehouse rate by (client skill, price signal) --
    s = tk["side"][live]
    px = s * (2 * dec["p_up"][live] - 1)      # price model agrees with client (+) / disagrees (-)
    ce = dec["q_used"][live]                  # estimated client skill
    wh = dec["warehouse"][live].astype(float)
    # quantile bin edges -- the spread of both signals moves a lot with the
    # sliders, so fixed edges would leave most of the grid empty.
    NX, NY = 11, 11

    def _edges(v: np.ndarray, n: int) -> np.ndarray:
        e = np.quantile(v, np.linspace(0, 1, n + 1))
        e[0], e[-1] = e[0] - 1e-9, e[-1] + 1e-9
        return np.maximum.accumulate(e + np.arange(n + 1) * 1e-12)

    xb, yb = _edges(ce, NX), _edges(px, NY)
    ix = np.clip(np.digitize(ce, xb) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(px, yb) - 1, 0, NY - 1)
    flat = iy * NX + ix
    cnt = np.bincount(flat, minlength=NX * NY)
    tot = np.bincount(flat, weights=wh, minlength=NX * NY)
    grid = [None if c < 8 else round(float(t / c), 3) for t, c in zip(tot, cnt)]
    surface = {"nx": NX, "ny": NY,
               "xedges": [round(float(v), 5) for v in xb],
               "yedges": [round(float(v), 5) for v in yb],
               "cells": grid}

    # ---- the tape: one ticket at a time, in time order --------------------
    tape = _tape(p, tk, dec, live)

    out = {
        "tape": tape,
        "stats": res["stats"],
        "curves": curves,
        "n_days_live": int(len(res["books"]["strategy"]["daily"])),
        "population": pop,
        "client_model": {"scatter": scatter, "corr": corr,
                         "prior_mean": cm["prior_mean"], "prior_conc": cm["prior_conc"]},
        "price_model": {"auc_train": pm["auc_train"], "auc_test": pm["auc_test"]},
        "surface": surface,
        "n_tickets_live": int(live.sum()),
        "params": {k: v for k, v in vars(p).items()},
    }
    if with_sweep:
        grid_pw = np.linspace(0.05, 0.95, 10)
        sw = sweep(p, grid_pw)
        out["sweep"] = [
            {"p_win": r["p_win"],
             "total": r["strategy"]["total"],
             "sharpe": r["strategy"]["sharpe"],
             "base_total": r["always_hedge"]["total"],
             "never_total": r["never_hedge"]["total"],
             "wh_rate": r["strategy"]["wh_rate"]}
            for r in sw
        ]
    return out


@app.post("/api/simulate")
def simulate(req: Req) -> dict[str, Any]:
    d = req.model_dump()
    with_sweep = d.pop("with_sweep")
    res = run(Params(**d))
    return _payload(res, with_sweep)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8090)))
