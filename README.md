# Hedge or warehouse — a two-model dealer desk

Interactive demo. A dealer sees pair-trade orders from a population of clients and makes
one decision per ticket: **hedge** it (go flat, pay the cost of going to market) or
**warehouse** it (take the other side and win exactly what the client loses).

Pure risk-warehousing economics — no client-spread revenue is modelled — so every number
on the page is edge over a desk that runs flat.

```
hedge      ->  PnL = -hedge_cost_bps * notional
warehouse  ->  PnL = -r_client * notional
```

## The setup

A ticket is a pair trade collapsed to a single spread return. Client *c* picks a side
`s ∈ {+1,-1}` and realises `r = s·x`, so **skill is side-selection ability**:

```
P(s == sign x) = 0.5 + q_c        =>   E[r_c] = 2·q_c·E|x|
```

Different mean per client, essentially the same variance — the population the brief asks
for. The variance-regime toggle switches per-instrument σ between common and dispersed.

## Two models, both actually fitted

| | what it predicts | how |
|---|---|---|
| **Price model** | `P(x > 0)` — direction of the underlying | `sklearn` `LogisticRegression` on noisy views of the latent driver plus a momentum feature and two decoys. Trained on the burn-in split only; the reported AUC is out-of-sample. |
| **Client model** | `q_c` — is this client any good | Hierarchical Beta-Binomial on the client's realised win/loss history. Prior fit by empirical Bayes (method of moments, sampling noise stripped out) on the burn-in window, then expanded **walk-forward** so a client the desk has barely traded stays shrunk to the population mean. |

## The decision

The client's own side is itself evidence about `x`, weighted by how sharp we think they
are. The two models combine by Bayes:

```
P(up | model, s, q̂)  ∝  P_model(up) · P(s | up, q̂)
E[r] / E|x|          =  s · (2·P(up | ·) − 1)
warehouse            iff  −E[r] > hedge_cost
```

Set `q̂ = 0` and it collapses to the price model alone; set `P_model = 0.5` and it
collapses to the client model alone. Both counterfactuals are computed and shown in the
attribution panel, so you can see what each model is worth on its own versus together.

The resulting decision surface is a clean diagonal frontier: 100% warehoused when the
client looks weak *and* the price model disagrees with their side, 0% when they look sharp
*and* the price model agrees.

## The tape

The last section is a scrubbing timer over the live window. At any moment it shows, for
that one ticket:

- **Model 1 output** — `P(underlying up)`
- **Model 2 output** — `P(this client wins)`, i.e. `0.5 + q̂`
- **Combined** — `P(client is on the right side)`, the posterior that drives the call
- the decision that follows, then the realised spread move and client return
- the desk's P&L on that ticket, what always-hedging would have made, and the **gain vs
  benchmark** attributable to that single decision

Plus a cumulative gain-over-benchmark line with a marker at the current moment. Scrubbing
is entirely client-side — it never refetches — and the cumulative line is exact over every
ticket even when the timer steps through an evenly-spaced sample (capped at 1,500 stops so
the payload stays light). It opens on the first ticket the desk warehouses, which is the
first moment the strategy differs from the benchmark at all.

## Layout

```
app/sim.py            simulation, both models, the decision rule, the book
app/server.py         FastAPI: POST /api/simulate, GET /api/health, serves the page
app/static/index.html the whole front end (no build step, no CDN)
```

The front end is dependency-free: hand-rolled SVG charts, hover tooltips, light/dark
themes, and a table view behind every chart. `?theme=light` / `?theme=dark` pins the
palette. All chart colors come from a validated categorical palette (CVD-checked in both
modes); direct end-labels and the table views are the relief channel for the one
light-mode slot that sits under 3:1 contrast.

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app.server:app --port 8090
# http://localhost:8090/
```

A full simulation — world, both model fits, the book, plus a ten-point re-run across the
whole winner-fraction axis — is ~120 ms, so the sliders are debounced rather than
precomputed.

## Deployment

Runs on the GCP box `gcpcai` (`predictnow-ai-cai-2026`, project `cai-api-504114`) under
systemd as `dealer-demo.service`, uvicorn on port **8090** with 2 workers.

```bash
sudo systemctl status dealer-demo
sudo systemctl restart dealer-demo
journalctl -u dealer-demo -f
```

To redeploy after local changes:

```bash
tar czf - --exclude='.venv' --exclude='__pycache__' --exclude='.git' -C .. dealer_presentation \
  | ssh gcpcai 'tar xzf - -C ~/'
ssh gcpcai 'sudo systemctl restart dealer-demo'
```

Live at **http://136.116.125.122:8090/** (`?theme=light` / `?theme=dark` to pin the palette).

### Firewall

Already open — rule `allow-dealer-demo-8090` (tcp:8090, `0.0.0.0/0`, target tag
`cai-portal`), mirroring the existing `allow-cai-8080`. It was created with:

```bash
gcloud compute firewall-rules create allow-dealer-demo-8090 \
  --project=cai-api-504114 --network=default \
  --allow=tcp:8090 --source-ranges=0.0.0.0/0 \
  --target-tags=cai-portal \
  --description="Dealer hedge-vs-warehouse demo"
```

Note the VM's own service account lacks compute scopes, so firewall changes must be run
from an authenticated gcloud elsewhere — on the Mac that is `~/google-cloud-sdk/bin/gcloud`
(not on PATH), authed as `ruchir@predictnow.ai`.

The port is open to the whole internet and the app has no auth in front of it. To take it
down: `gcloud compute firewall-rules delete allow-dealer-demo-8090`.

An SSH tunnel also works without any firewall rule:

```bash
ssh -N -L 8091:localhost:8090 gcpcai   # then http://localhost:8091/
```

## Caveats

- Everything is synthetic. The point is the mechanism and the shape of the tradeoff, not
  a calibrated P&L forecast.
- Warehoused risk carries no capital charge and no inventory limit, so the strategy's
  Sharpe is flattering in absolute terms. Read it against the always-hedge and never-hedge
  lines, not on its own.
- The client-skill prior is a single Beta even though the true population is a two-component
  mixture. That is a deliberate, honest approximation — the estimator does not get told the
  generating structure.
