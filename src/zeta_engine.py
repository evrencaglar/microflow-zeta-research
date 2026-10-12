"""Microflow zeta research engine — Python reference implementation.

Core analytics of the Microflow Scalper tick-flow research engine,
re-implemented in Python for transparency. See README.md for methodology.

Author: Evren Caglar (The Next Tick Algo Systems)
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

UP, DOWN, FLAT = 1, -1, 0


# --------------------------------------------------------------------------- #
# Tick representation and L1 midpoint-delta classifier
# --------------------------------------------------------------------------- #
@dataclass
class Tick:
    time_ms: int          # epoch milliseconds
    bid: float
    ask: float

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0


class MidpointDeltaClassifier:
    """Classifies each tick as up/down/flat against the previous spread
    midpoint. Classifying on the midpoint (not the bid alone) removes the
    one-sided bias of quote-driven feeds."""

    def __init__(self) -> None:
        self._prev_mid: Optional[float] = None
        self.last_side: int = FLAT

    def classify(self, tick: Tick) -> int:
        mid = tick.midpoint
        if self._prev_mid is None:
            self._prev_mid = mid
            self.last_side = FLAT
            return FLAT
        if mid > self._prev_mid:
            side = UP
        elif mid < self._prev_mid:
            side = DOWN
        else:
            side = FLAT
        self._prev_mid = mid
        self.last_side = side
        return side


# --------------------------------------------------------------------------- #
# Zeta bars (time-based micro-bars over the confirmed tick sequence)
# --------------------------------------------------------------------------- #
@dataclass
class Zeta:
    time_start_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    up_ticks: int = 0
    down_ticks: int = 0
    flat_ticks: int = 0

    @property
    def distance(self) -> float:          # body
        return abs(self.close - self.open)

    @property
    def range(self) -> float:             # high-low
        return self.high - self.low

    @property
    def z_ratio(self) -> float:           # effort / result
        return np.inf if self.distance == 0 else self.range / self.distance

    @property
    def tick_delta(self) -> int:          # signed
        return self.up_ticks - self.down_ticks

    @property
    def delta_ratio_pct(self) -> float:   # signed, % of volume
        return 0.0 if self.volume == 0 else self.tick_delta / self.volume * 100.0

    @property
    def efficiency(self) -> float:        # distance per unit volume
        return 0.0 if self.volume == 0 else self.distance / self.volume

    @property
    def type(self) -> int:                # BUY / SELL / NEUTRAL
        d = self.tick_delta
        return UP if d > 0 else (DOWN if d < 0 else FLAT)

    def to_row(self) -> dict:
        return {
            "time_start_ms": self.time_start_ms,
            "open": self.open, "high": self.high, "low": self.low,
            "close": self.close, "volume": self.volume,
            "up": self.up_ticks, "down": self.down_ticks, "flat": self.flat_ticks,
            "tick_delta": self.tick_delta,
            "delta_ratio_pct": round(self.delta_ratio_pct, 2),
            "distance": round(self.distance, 5),
            "range": round(self.range, 5),
            "z_ratio": round(self.z_ratio, 2),
            "efficiency": round(self.efficiency, 8),
            "type": {UP: "BUY", DOWN: "SELL", FLAT: "NEUTRAL"}[self.type],
        }


class ZetaBuilder:
    """Accumulates classified ticks into time-based zeta bars.

    A zeta closes when a tick arrives at or after the next N-second boundary.
    Only confirmed ticks enter; no provisional quotes advance the state.
    """

    def __init__(self, seconds: int, classifier: MidpointDeltaClassifier) -> None:
        self.seconds = seconds
        self.cls = classifier
        self._cur: Optional[Zeta] = None

    def update(self, tick: Tick) -> Optional[Zeta]:
        side = self.cls.classify(tick)
        closed: Optional[Zeta] = None
        if self._cur is not None and tick.time_ms >= self._cur.time_start_ms + self.seconds * 1000:
            closed = self._cur
            self._cur = None
        if self._cur is None:
            self._cur = Zeta(time_start_ms=(tick.time_ms // (self.seconds * 1000)) * self.seconds * 1000,
                             open=tick.midpoint, high=tick.midpoint,
                             low=tick.midpoint, close=tick.midpoint)
        z = self._cur
        z.close = tick.midpoint
        z.high = max(z.high, tick.midpoint)
        z.low = min(z.low, tick.midpoint)
        z.volume += 1
        if side == UP:
            z.up_ticks += 1
        elif side == DOWN:
            z.down_ticks += 1
        else:
            z.flat_ticks += 1
        return closed


# --------------------------------------------------------------------------- #
# Research context zeta (wide rolling window, reset at session open)
# --------------------------------------------------------------------------- #
class ContextZeta:
    """Rolling tick-count context window (default 1000 ticks).

    Maintains the wider flow-regime features used by the research profiles:
    delta-ratio z-score and the kernel-weighted pressure score pi with its
    z-score pi_z. Resets at the first confirmed tick at/after the configured
    session-open time (16:30 broker time for the NY-open research).
    """

    def __init__(self, n_ticks: int = 1000, session_open_hhmm: int = 1630,
                 kernel_half_life: float = 0.25) -> None:
        self.n = n_ticks
        self.session_open = session_open_hhmm
        self.hl = kernel_half_life
        self._deltas: list[float] = []
        self._ratios: list[float] = []
        self._last_reset_day: Optional[int] = None

    def _maybe_reset(self, tick: Tick) -> None:
        day_ms = 86_400_000
        day_start = (tick.time_ms // day_ms) * day_ms
        mins = (tick.time_ms - day_start) // 60_000
        hhmm = int(mins // 60) * 100 + int(mins % 60)
        day_id = tick.time_ms // day_ms
        if hhmm >= self.session_open and self._last_reset_day != day_id:
            self._deltas.clear()
            self._ratios.clear()
            self._last_reset_day = day_id

    def update(self, tick: Tick, side: int, window_volume: int) -> dict:
        self._maybe_reset(tick)
        if side == FLAT or window_volume == 0:
            return self.snapshot()
        self._deltas.append(float(side))
        self._ratios.append(float(side) / window_volume)
        if len(self._deltas) > self.n:
            self._deltas.pop(0)
            self._ratios.pop(0)
        return self.snapshot()

    def snapshot(self) -> dict:
        if not self._deltas:
            return {"delta_ratio_z": 0.0, "pi": 0.0, "pi_z": 0.0, "n": 0}
        ratios = np.asarray(self._ratios)
        # kernel-weighted pressure: recent zetas weigh more (half-life param)
        w = 0.5 ** (np.arange(len(ratios))[::-1] / max(1.0, self.hl * len(ratios)))
        pi = float(np.dot(w, ratios) / w.sum())
        mu, sd = float(ratios.mean()), float(ratios.std(ddof=0))
        return {
            "delta_ratio_z": 0.0 if sd == 0 else float((ratios[-1] - mu) / sd),
            "pi": pi,
            "pi_z": 0.0 if sd == 0 else float((pi - mu) / sd),
            "n": len(ratios),
        }


# --------------------------------------------------------------------------- #
# Push-alignment entry gate (Engine 1, simplified)
# --------------------------------------------------------------------------- #
@dataclass
class GateConfig:
    min_abs_delta_ratio_pct: float = 10.0   # conviction floor
    band_veto_lo: float = 25.0              # block mid-conviction pushes
    band_veto_hi: float = 60.0
    max_z_ratio: float = 6.0                # block absorption-fighting zetas


@dataclass
class GateDecision:
    side: int
    allowed: bool
    reasons: list[str] = field(default_factory=list)


def push_alignment_gate(z: Zeta, ctx: dict, cfg: GateConfig) -> GateDecision:
    """Long only when zeta direction AND signed tick-delta agree; short
    symmetrically. Band vetoes block the absorption regime where naive
    momentum gets trapped."""
    side = z.type
    if side == FLAT:
        return GateDecision(FLAT, False, ["neutral zeta"])
    dr = abs(z.delta_ratio_pct)
    reasons: list[str] = []
    if dr < cfg.min_abs_delta_ratio_pct:
        reasons.append(f"weak conviction ({dr:.1f}% < {cfg.min_abs_delta_ratio_pct}%)")
    if cfg.band_veto_lo < dr <= cfg.band_veto_hi:
        reasons.append(f"mid-conviction band veto ({dr:.1f}%)")
    if z.z_ratio > cfg.max_z_ratio:
        reasons.append(f"absorption (z_ratio {z.z_ratio:.1f} > {cfg.max_z_ratio})")
    # context agreement: do not fight an established opposite pressure regime
    if ctx.get("pi_z", 0.0) * side < -2.0:
        reasons.append(f"context pressure against ({ctx['pi_z']:.2f})")
    return GateDecision(side, not reasons, reasons)
