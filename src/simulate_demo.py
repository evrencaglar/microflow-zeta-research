"""Synthetic-tick demo for the Microflow zeta research engine.

Generates a two-sided synthetic quote stream with an injected one-sided
burst, builds zeta bars, and prints every push-alignment evaluation.
Validates the plumbing only -- it does not demonstrate any trading edge.
"""
import numpy as np
import pandas as pd

from zeta_engine import (ContextZeta, GateConfig, MidpointDeltaClassifier,
                         Tick, UP, ZetaBuilder, push_alignment_gate)


def synthetic_ticks(n: int = 6000, seed: int = 7) -> list[Tick]:
    rng = np.random.default_rng(seed)
    mid = 100.0
    ticks: list[Tick] = []
    t = 1_700_000_000_000  # fixed epoch ms for reproducibility
    for i in range(n):
        # regime: two-sided noise, except a one-sided buy burst in the middle
        in_burst = 2500 <= i < 2900
        drift = 0.0006 if in_burst else 0.0
        mid += rng.normal(drift, 0.0004)
        spread = 0.0002
        bid, ask = mid - spread / 2, mid + spread / 2
        t += int(rng.integers(80, 220))  # irregular inter-tick times
        ticks.append(Tick(time_ms=t, bid=round(bid, 5), ask=round(ask, 5)))
    return ticks


def main() -> None:
    ticks = synthetic_ticks()
    cls = MidpointDeltaClassifier()
    builder = ZetaBuilder(seconds=10, classifier=cls)      # trigger zeta
    ctx = ContextZeta(n_ticks=1000, session_open_hhmm=0)   # no session cut in demo
    cfg = GateConfig()

    zetas, decisions = [], []
    for tick in ticks:
        closed = builder.update(tick)
        if closed is not None:
            snap = ctx.snapshot()
            dec = push_alignment_gate(closed, snap, cfg)
            zetas.append(closed.to_row())
            if dec.side != 0:
                decisions.append({
                    "time_start_ms": closed.time_start_ms,
                    "side": {UP: "LONG", -1: "SHORT"}[dec.side],
                    "allowed": dec.allowed,
                    "reasons": "; ".join(dec.reasons) if dec.reasons else "-",
                    "pi_z": round(snap["pi_z"], 2),
                })
        ctx.update(tick, cls.last_side, builder._cur.volume if builder._cur else 0)

    zdf = pd.DataFrame(zetas)
    print(f"closed zetas: {len(zdf)} | avg volume/zeta: {zdf['volume'].mean():.0f}")
    print("\nbuy-vs-sell balance (tick_delta):")
    print(zdf.groupby("type")["volume"].agg(["count", "sum"]))
    print("\npush-alignment evaluations (non-neutral zetas):")
    ddf = pd.DataFrame(decisions)
    print(ddf.to_string(index=False, max_rows=40))
    print(f"\nallowed signals: {int(ddf['allowed'].sum()) if len(ddf) else 0} "
          f"of {len(ddf)} evaluated")


if __name__ == "__main__":
    main()
