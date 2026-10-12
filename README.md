# Microflow Zeta Research Engine (Python reference implementation)

A clean Python reference implementation of the core analytics from the
**Microflow Scalper** tick-flow research engine (MQL5, MT5), published for
methodological transparency and reproducibility.

Author: **Evren Caglar** — The Next Tick Algo Systems (Est. 2023)
License: MIT (this reference implementation). The production MQL5 EA remains
proprietary and is not part of this repository.

## What this repository is

This is the *research core* of a tick-flow scalping system, re-implemented in
Python so that the methodology can be read, tested and criticised without an
MT5 terminal. It is **not** a port of the trading EA: order routing, position
management, session safeguards and the full filter stack are deliberately
excluded.

## Methodology

### 1. Tick classification (L1 midpoint-delta classifier)

Each incoming quote updates the spread midpoint `m = (bid + ask) / 2`. A tick
is classified against the previous midpoint:

- `m > m_prev`  → up-tick
- `m < m_prev`  → down-tick
- `m == m_prev` → flat-tick

Classifying on the *midpoint* rather than the bid alone removes the one-sided
bias of quote-driven feeds: both sides of the spread inform the classification.

### 2. Zeta bars (time-based micro-bars)

A **zeta** is a time-based bar of `N` seconds built from the *confirmed* tick
sequence (no provisional quotes). Per closed zeta we record:

| Field | Definition |
|---|---|
| `open/high/low/close` | quote midpoints at zeta boundaries |
| `volume` | tick count |
| `up/down/flat_ticks` | classifier counts |
| `tick_delta` | `up_ticks - down_ticks` (signed) |
| `delta_ratio_pct` | `tick_delta / volume * 100` |
| `distance` | `\|close - open\|` |
| `range` | `\|high - low\|` |
| `z_ratio` | `range / distance` (effort vs. result) |
| `efficiency` | `distance / volume` |

`z_ratio` is the effort/result ratio: a large `range` with a small body means
two-sided fighting (absorption); a small `range` with a large body means
one-sided conviction (displacement).

### 3. Dual-scale research context (v8.19)

Two independent zeta streams are built from the same confirmed tick sequence:

- **Trigger zeta** (default 300 ticks / "Core") — the execution-relevant scale.
- **Context zeta** (default 1000 ticks) — a wider window whose rolling
  features (delta-ratio z-score, kernel-weighted pressure `pi` and its
  z-score `pi_z`) describe the flow regime. No partial or future context is
  consumed: a trigger decision sees only the *latest completed* context zeta.

Both streams reset at the first confirmed tick at/after 16:30 broker time
(NY equity open), so session carry-over cannot contaminate the research state.

### 4. Push-alignment entry gate

The primary entry condition is *directional agreement*: a long trigger is
considered only when the trigger zeta direction **and** the signed tick-delta
direction agree (short symmetrically). Around this core, band filters on
`|delta_ratio_pct|` and `z_ratio` block mid-conviction pushes — the "someone
is absorbing this move" regime where naive momentum entries get trapped.

### 5. Honest exits (research profile)

`1 ATR` protective stop → scale 50% at `2 ATR` → break-even → final 50% at
`3 ATR`, with a hard 30-minute maximum holding period. Exits are tagged with
their true reason (stop, target, break-even, time) rather than the flattering
one.

## Repository layout

```
src/zeta_engine.py     # tick classifier, zeta builder, context zeta, entry gate
src/simulate_demo.py   # synthetic-tick demo: builds zetas, prints a signal summary
```

## Running the demo

```bash
pip install numpy pandas
python src/simulate_demo.py
```

The demo generates a synthetic two-sided tick sequence with an injected
one-sided burst and prints the closed zetas plus every push-alignment
evaluation — no market data or terminal required.

## What is intentionally NOT here

- The production MQL5 EA (`Microflow_Scalper_MT5`) and its execution layer
- Any tuned threshold values from the live research profiles (thresholds in
  this repository are illustrative defaults; the research values remain
  private while they are traded)
- Broker-specific handling (coalesced `NewTick` catch-up, spread session
  guards, slippage controls)

## Honest limitations

- The synthetic demo validates the *plumbing*, not edge. The April–August
  2026 US100 first-two-hours research that produced the CORE/STRICT profiles
  used real broker tick data and is not reproduced here.
- Zeta boundaries are time-based; sub-ms arrival ordering within a zeta is
  feed-dependent and ignored.

## Disclaimer

Published for research and educational purposes. Nothing here is investment
advice or a recommendation to trade.
