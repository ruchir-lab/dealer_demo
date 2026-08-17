"""
Dealer hedge-vs-warehouse simulation.

Economics (pure risk warehousing -- no client-spread revenue):

    hedge      ->  dealer PnL = -hedge_cost_bps * notional      (flat, tiny drag)
    warehouse  ->  dealer PnL = -r_client * notional            (dealer is the other side)

Each ticket is a *pair trade*, collapsed to a single spread return r.

Generative story
----------------
Instrument k has a forward spread return x[k,t]. A client picks a side
s in {+1,-1}; their realized return is r = s * x. Client skill is *side-
selection ability*:

    P(s == sign(x)) = 0.5 + q_c ,   q_c in (-0.5, 0.5)

so E[r_c] = 2 * q_c * E|x|  -- different mean per client, essentially the
same variance (Var[r] = Var[x]), exactly the population the brief asks for.
The variance regime knob scales per-instrument sigma so variance can be
common or dispersed.

Two models, both actually fitted
--------------------------------
1. PRICE MODEL   sklearn LogisticRegression on noisy views of the latent
                 driver -> P(x > 0). Trained on the train split only.
2. CLIENT MODEL  hierarchical Beta-Binomial on each client's realised
                 win/loss history, prior fit by empirical Bayes (method of
                 moments) on the burn-in window -> posterior mean skill
                 q_hat. Expanded walk-forward, so a client the desk has
                 barely traded stays shrunk to the population mean.

Decision
--------
The client's own side is *itself* evidence about x, weighted by how sharp
we think they are. Bayes:

    P(up | model, s, q) ~ P_model(up) * P(s | up, q)

    E[r] / E|x| = s * ( P(up | .) - P(dn | .) )

    warehouse  iff  -E[r] > hedge_cost      (expected warehouse edge beats
                                             the cost of hedging)

q_hat -> 0 collapses this to the price model alone; P_model -> 0.5
collapses it to the client model alone. That decomposition is what the
"edge attribution" panel reports.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BPS = 1e-4


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

@dataclass
class Params:
    # --- population of clients -------------------------------------------
    p_win: float = 0.35          # fraction of clients that are genuinely skilled
    skill_edge: float = 0.10     # |q| separation between winners and losers
    skill_disp: float = 0.04     # within-group dispersion of q
    n_clients: int = 60

    # --- variance regime --------------------------------------------------
    vol_regime: str = "common"   # "common" | "random"
    vol_disp: float = 0.5        # dispersion of per-instrument sigma when random
    base_vol_bps: float = 90.0   # typical per-ticket spread move, bps

    # --- price predictability --------------------------------------------
    price_snr: float = 0.35      # 0 = unpredictable, 1 = strongly predictable

    # --- desk / book ------------------------------------------------------
    n_days: int = 500
    tickets_per_day: int = 12
    n_instruments: int = 8
    notional_mm: float = 5.0     # median ticket notional, $mm
    notional_disp: float = 0.6
    hedge_cost_bps: float = 0.5  # cost of taking the hedge to market
    train_frac: float = 0.40     # burn-in: price model train + client prior

    seed: int = 7

    def clean(self) -> "Params":
        self.p_win = float(np.clip(self.p_win, 0.0, 1.0))
        self.skill_edge = float(np.clip(self.skill_edge, 0.0, 0.30))
        self.skill_disp = float(np.clip(self.skill_disp, 0.0, 0.15))
        self.price_snr = float(np.clip(self.price_snr, 0.0, 1.0))
        self.vol_disp = float(np.clip(self.vol_disp, 0.0, 1.5))
        self.hedge_cost_bps = float(np.clip(self.hedge_cost_bps, 0.0, 10.0))
        self.n_clients = int(np.clip(self.n_clients, 10, 400))
        self.n_days = int(np.clip(self.n_days, 120, 2000))
        self.tickets_per_day = int(np.clip(self.tickets_per_day, 2, 60))
        self.n_instruments = int(np.clip(self.n_instruments, 2, 40))
        self.train_frac = float(np.clip(self.train_frac, 0.15, 0.7))
        self.vol_regime = "random" if str(self.vol_regime).lower() == "random" else "common"
        return self


# --------------------------------------------------------------------------
# world generation
# --------------------------------------------------------------------------

def _build_world(p: Params, rng: np.random.Generator) -> dict[str, Any]:
    """Synthetic instruments, their latent driver, observable features, and
    the realised forward spread return x[k, t]."""
    K, T = p.n_instruments, p.n_days

    # latent AR(1) driver, unit variance
    phi = 0.72
    shocks = rng.standard_normal((K, T)) * np.sqrt(1 - phi ** 2)
    lat = np.empty((K, T))
    lat[:, 0] = rng.standard_normal(K)
    for t in range(1, T):
        lat[:, t] = phi * lat[:, t - 1] + shocks[:, t]

    # per-instrument vol
    if p.vol_regime == "random":
        sig = p.base_vol_bps * BPS * np.exp(p.vol_disp * rng.standard_normal(K)
                                            - 0.5 * p.vol_disp ** 2)
    else:
        sig = np.full(K, p.base_vol_bps * BPS)

    # how much of the forward move the driver explains
    rho = 0.62 * p.price_snr                       # correlation(latent, x)
    x = (rho * lat + np.sqrt(max(1e-9, 1 - rho ** 2)) * rng.standard_normal((K, T)))
    x = x * sig[:, None]

    # observable features: three noisy views of the driver + two decoys.
    # noise shrinks as price_snr rises, so the fitted model's AUC really moves.
    view_noise = 0.55 + 2.6 * (1.0 - p.price_snr)
    feats = [lat + view_noise * rng.standard_normal((K, T)) for _ in range(3)]
    feats += [rng.standard_normal((K, T)) for _ in range(2)]
    # a genuine momentum feature computed from the observable price path
    mom = np.zeros((K, T))
    mom[:, 1:] = np.cumsum(x, axis=1)[:, :-1] / (sig[:, None] * np.sqrt(np.arange(1, T)))
    feats.append(np.nan_to_num(mom))

    F = np.stack(feats, axis=-1)                   # (K, T, n_feat)
    return {"x": x, "F": F, "sigma": sig, "latent": lat}


def _build_tickets(p: Params, world: dict, rng: np.random.Generator) -> dict[str, Any]:
    """Client population, their orders, and the realised client return."""
    N = p.n_clients
    n_tk = p.n_days * p.tickets_per_day

    # --- client skills: two-component mixture -----------------------------
    is_win = rng.random(N) < p.p_win
    q = np.where(is_win,
                 p.skill_edge + p.skill_disp * rng.standard_normal(N),
                 -p.skill_edge + p.skill_disp * rng.standard_normal(N))
    q = np.clip(q, -0.45, 0.45)

    # heavier clients trade more often
    activity = rng.gamma(2.0, 1.0, N)
    activity /= activity.sum()

    day = np.repeat(np.arange(p.n_days), p.tickets_per_day)
    cli = rng.choice(N, size=n_tk, p=activity)
    ins = rng.integers(0, p.n_instruments, size=n_tk)

    notional = p.notional_mm * np.exp(p.notional_disp * rng.standard_normal(n_tk)
                                      - 0.5 * p.notional_disp ** 2)

    x_tk = world["x"][ins, day]
    right = rng.random(n_tk) < (0.5 + q[cli])      # did the client pick the winning side?
    side = np.where(right, np.sign(x_tk), -np.sign(x_tk))
    side[side == 0] = 1.0
    r = side * x_tk                                 # client return on the pair

    return {
        "day": day, "cli": cli, "ins": ins, "side": side, "notional": notional,
        "x": x_tk, "r": r, "won": (r > 0).astype(int),
        "q_true": q, "is_win": is_win, "activity": activity,
    }


# --------------------------------------------------------------------------
# model 1 -- price direction
# --------------------------------------------------------------------------

def _fit_price_model(p: Params, world: dict, tk: dict, t_split: int) -> dict[str, Any]:
    K, T, nf = world["F"].shape
    F = world["F"].reshape(-1, nf)
    y = (world["x"].reshape(-1) > 0).astype(int)
    dayg = np.tile(np.arange(T), K)

    tr, te = dayg < t_split, dayg >= t_split
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-12
    Z = (F - mu) / sd

    clf = LogisticRegression(C=0.35, max_iter=400)
    clf.fit(Z[tr], y[tr])
    prob = clf.predict_proba(Z)[:, 1].reshape(K, T)

    auc_tr = roc_auc_score(y[tr], clf.predict_proba(Z[tr])[:, 1])
    auc_te = roc_auc_score(y[te], clf.predict_proba(Z[te])[:, 1]) if te.sum() > 10 else float("nan")

    return {"prob": prob, "auc_train": float(auc_tr), "auc_test": float(auc_te),
            "p_up_ticket": prob[tk["ins"], tk["day"]]}


# --------------------------------------------------------------------------
# model 2 -- client skill (hierarchical Beta-Binomial, walk-forward)
# --------------------------------------------------------------------------

def _fit_client_model(p: Params, tk: dict, t_split: int) -> dict[str, Any]:
    N = p.n_clients
    cli, won, day = tk["cli"], tk["won"], tk["day"]

    # --- empirical-Bayes prior from the burn-in window --------------------
    burn = day < t_split
    n_c = np.bincount(cli[burn], minlength=N).astype(float)
    w_c = np.bincount(cli[burn], weights=won[burn], minlength=N).astype(float)
    seen = n_c >= 8
    if seen.sum() >= 5:
        rates = w_c[seen] / n_c[seen]
        m, v = float(rates.mean()), float(rates.var(ddof=1))
        n_bar = float(n_c[seen].mean())
        # strip binomial sampling noise to get the true between-client variance
        v_between = max(v - m * (1 - m) / max(n_bar, 1.0), 1e-5)
        conc = max(m * (1 - m) / v_between - 1.0, 2.0)
    else:
        m, conc = 0.5, 40.0
    a0, b0 = m * conc, (1 - m) * conc

    # --- walk-forward posterior -------------------------------------------
    n_tk = len(cli)
    a = np.full(N, a0)
    b = np.full(N, b0)
    q_hat = np.empty(n_tk)
    n_seen = np.empty(n_tk, dtype=int)
    for i in range(n_tk):
        c = cli[i]
        q_hat[i] = a[c] / (a[c] + b[c]) - 0.5      # decided BEFORE this ticket resolves
        n_seen[i] = int(a[c] + b[c] - a0 - b0)
        if won[i]:
            a[c] += 1.0
        else:
            b[c] += 1.0

    q_final = a / (a + b) - 0.5
    return {"q_hat": q_hat, "q_final": q_final, "n_seen": n_seen,
            "prior_mean": m, "prior_conc": float(conc),
            "n_trades_final": (a + b - a0 - b0).astype(int)}


# --------------------------------------------------------------------------
# decision + book
# --------------------------------------------------------------------------

def _decide(p: Params, tk: dict, pm: dict, cm: dict, world: dict) -> dict[str, Any]:
    """Bayesian combination -> expected client return -> warehouse flag."""
    p_up = np.clip(pm["p_up_ticket"], 1e-4, 1 - 1e-4)
    q = np.clip(cm["q_hat"], -0.45, 0.45)
    s = tk["side"]

    # P(side | direction, skill)
    lik_up = np.where(s > 0, 0.5 + q, 0.5 - q)
    lik_dn = np.where(s > 0, 0.5 - q, 0.5 + q)

    post_up = p_up * lik_up
    post_dn = (1 - p_up) * lik_dn
    post_up = post_up / (post_up + post_dn + 1e-12)

    e_abs_x = world["sigma"][tk["ins"]] * np.sqrt(2 / np.pi)
    e_r = s * (2 * post_up - 1) * e_abs_x           # expected client return
    warehouse = (-e_r) > (p.hedge_cost_bps * BPS)

    # counterfactual rules for edge attribution
    pu_price = p_up
    e_r_price = s * (2 * pu_price - 1) * e_abs_x
    wh_price = (-e_r_price) > (p.hedge_cost_bps * BPS)

    lu, ld = lik_up, lik_dn
    pu_cli = 0.5 * lu / (0.5 * lu + 0.5 * ld + 1e-12)
    e_r_cli = s * (2 * pu_cli - 1) * e_abs_x
    wh_cli = (-e_r_cli) > (p.hedge_cost_bps * BPS)

    return {"warehouse": warehouse, "e_r": e_r, "post_up": post_up,
            "p_up": p_up, "q_used": q,
            "wh_price": wh_price, "wh_client": wh_cli}


def _book(p: Params, tk: dict, warehouse: np.ndarray, live: np.ndarray) -> dict[str, Any]:
    """Daily PnL in $ for one decision rule, over the live window."""
    notion = tk["notional"][live] * 1e6
    r = tk["r"][live]
    day = tk["day"][live]
    wh = warehouse[live].astype(bool)

    pnl = np.where(wh, -r * notion, -p.hedge_cost_bps * BPS * notion)
    d0 = day.min()
    nd = int(day.max() - d0 + 1)
    daily = np.bincount(day - d0, weights=pnl, minlength=nd)
    risk = np.bincount(day - d0, weights=np.where(wh, notion, 0.0), minlength=nd)
    return {"daily": daily, "risk": risk, "wh_rate": float(wh.mean())}


def _stats(daily: np.ndarray, risk: np.ndarray, wh_rate: float) -> dict[str, Any]:
    mu, sd = float(daily.mean()), float(daily.std(ddof=1))
    cum = np.cumsum(daily)
    dd = cum - np.maximum.accumulate(cum)
    # a book that never warehouses has no risk -- its PnL is a deterministic
    # hedge-cost drag, so a Sharpe ratio for it is meaningless, not infinite.
    sharpe = None if (wh_rate <= 0.0 or sd <= 1e-9) else float(mu / sd * np.sqrt(252))
    return {
        "mean_daily": mu,
        "total": float(cum[-1]),
        "vol_daily": sd,
        "sharpe": sharpe,
        "hit_rate": float((daily > 0).mean()),
        "var95": float(np.percentile(daily, 5)),
        "worst_day": float(daily.min()),
        "max_dd": float(dd.min()),
        "capital_at_risk": float(risk.mean()),
        "peak_risk": float(risk.max()),
        "wh_rate": wh_rate,
    }


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

def _core(p: Params, world: dict, tk: dict) -> dict[str, Any]:
    t_split = int(p.n_days * p.train_frac)
    live = tk["day"] >= t_split

    pm = _fit_price_model(p, world, tk, t_split)
    cm = _fit_client_model(p, tk, t_split)
    dec = _decide(p, tk, pm, cm, world)

    always = np.zeros(len(tk["r"]), dtype=bool)
    never = np.ones(len(tk["r"]), dtype=bool)
    oracle = (tk["r"] < 0)                          # perfect foresight upper bound

    rules = {
        "always_hedge": always,
        "strategy": dec["warehouse"],
        "never_hedge": never,
        "price_only": dec["wh_price"],
        "client_only": dec["wh_client"],
        "oracle": oracle,
    }
    books = {k: _book(p, tk, v, live) for k, v in rules.items()}
    stats = {k: _stats(b["daily"], b["risk"], b["wh_rate"]) for k, b in books.items()}
    return {"pm": pm, "cm": cm, "dec": dec, "books": books, "stats": stats,
            "live": live, "t_split": t_split}


def run(p: Params) -> dict[str, Any]:
    p = p.clean()
    rng = np.random.default_rng(p.seed)
    world = _build_world(p, rng)
    tk = _build_tickets(p, world, rng)
    return {"p": p, "world": world, "tk": tk, **_core(p, world, tk)}


def sweep(p: Params, grid: np.ndarray) -> list[dict[str, Any]]:
    """Re-run across the winner-fraction axis, everything else held fixed."""
    out = []
    for g in grid:
        q = Params(**{**asdict(p), "p_win": float(g)}).clean()
        rng = np.random.default_rng(q.seed)
        world = _build_world(q, rng)
        tk = _build_tickets(q, world, rng)
        c = _core(q, world, tk)
        out.append({"p_win": float(g),
                    "strategy": c["stats"]["strategy"],
                    "always_hedge": c["stats"]["always_hedge"],
                    "never_hedge": c["stats"]["never_hedge"]})
    return out
