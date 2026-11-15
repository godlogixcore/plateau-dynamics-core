#!/usr/bin/env python3
"""
Plateau Dynamics — standalone reference engine
Ruben Kabongo, 2026

Manuscript:
"Plateau Dynamics: A Unified Operational Definition for Real-Time Detection
and Measurement of Persistent Low-Dynamical Objects in Observation Space"
ORCID: 0009-0005-5573-0857

Constitutive sequence
---------------------
    f -> (d_m, v_m) -> C_m -> pi_0(C_m) -> Pi(P) >= pi_min
      -> Plateau -> O(P)

Design rules
------------
1. Plateau identity is issued only after bounded drift, bounded variation,
   connectedness/contiguity, and persistence are satisfied.
2. Accumulated Order is post-identity: it never creates Plateau membership.
3. The live engine is causal. It uses only the current and previous samples;
   there are no centered windows, future samples, bfill, or retrospective
   threshold changes in live operation.
4. A live threshold lens is frozen before detection. If thresholds are not
   supplied, a calibration prefix proposes them once and then freezes them.
5. Live Order begins at confirmation and is never back-filled over the
   candidate waiting interval.
6. Current membership and retained accumulated Order are separate outputs.
7. Spatial Plateau construction uses the same constitution -> measurement
   order with declared adjacency.

Dependencies
------------
Required: numpy, pandas, scipy
Optional: healpy (only for HEALPix first-ring adjacency)

This file contains:
- Measurement contract (Lambda)
- Retrospective temporal constitution
- Fully causal live/streaming temporal engine
- File streaming and replay
- Generic spatial Plateau constitution
- HEALPix wrapper using official healpy neighbours
- Cross-scale lineage helper
- CSV/JSON export helpers
- Command line interface
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]
Adjacency = Union[Sequence[Sequence[int]], Mapping[int, Sequence[int]], Callable[[int], Sequence[int]]]


# =============================================================================
# Measurement contract
# =============================================================================

@dataclass(frozen=True)
class Lambda:
    """Declared PD measurement contract and implementation settings.

    Constitutive core:
        Lambda = (m, D_m, V_m, eps_d, eps_v, pi_min)

    The remaining fields specify how that contract is numerically realized.
    They do not change Plateau identity itself.
    """

    # Constitutive / local-operator contract
    m: int
    D: str = "slope"                  # slope | diff | savgol
    V: str = "rolling_var"            # rolling_var
    eps_d: Optional[float] = None
    eps_v: Optional[float] = None
    pi_min: Optional[float] = None

    # Threshold proposal (used only if eps_d / eps_v are absent)
    q_d: float = 0.40
    q_v: float = 0.40
    calibration_points: Optional[int] = None

    # Temporal realization
    persistence: str = "time"         # time | samples
    smooth_m: int = 1                  # 1 = no smoothing
    variation_signal: str = "raw"     # raw | smoothed
    sg_polyorder: int = 2
    require_full_window: bool = True

    # Post-identity Order
    rho: float = 1e-9

    def validate(self) -> "Lambda":
        if int(self.m) < 2:
            raise ValueError("m must be >= 2")
        if int(self.smooth_m) < 1:
            raise ValueError("smooth_m must be >= 1")
        if self.D not in {"slope", "diff", "savgol"}:
            raise ValueError("D must be one of: slope, diff, savgol")
        if self.V != "rolling_var":
            raise ValueError("V currently supports only rolling_var")
        if self.persistence not in {"time", "samples"}:
            raise ValueError("persistence must be 'time' or 'samples'")
        if self.variation_signal not in {"raw", "smoothed"}:
            raise ValueError("variation_signal must be 'raw' or 'smoothed'")
        if not (0.0 < float(self.q_d) < 1.0):
            raise ValueError("q_d must lie in (0,1)")
        if not (0.0 < float(self.q_v) < 1.0):
            raise ValueError("q_v must lie in (0,1)")
        if self.eps_d is not None and self.eps_d < 0:
            raise ValueError("eps_d must be >= 0")
        if self.eps_v is not None and self.eps_v < 0:
            raise ValueError("eps_v must be >= 0")
        if self.pi_min is not None and self.pi_min < 0:
            raise ValueError("pi_min must be >= 0")
        if self.rho <= 0:
            raise ValueError("rho must be > 0")
        return self


# =============================================================================
# Basic utilities
# =============================================================================

def _as_1d(x: ArrayLike, *, name: str = "array") -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return a


def _time(n: int, t: Optional[ArrayLike]) -> np.ndarray:
    if t is None:
        return np.arange(n, dtype=float)
    tt = _as_1d(t, name="t")
    if len(tt) != n:
        raise ValueError(f"t length {len(tt)} != signal length {n}")
    if n > 1 and np.any(np.diff(tt) <= 0):
        raise ValueError("t must be strictly increasing")
    return tt


def _finite_pair(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(t) & np.isfinite(y)
    return t[mask], y[mask]


def _safe_quantile(x: np.ndarray, q: float) -> float:
    z = np.asarray(x, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        raise ValueError("cannot compute threshold from an empty finite sample")
    out = float(np.quantile(z, q))
    if out <= 0:
        positive = z[z > 0]
        if len(positive):
            out = float(np.median(positive))
        else:
            out = 1e-12
    return out


def _rolling_mean_retrospective(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.astype(float, copy=True)
    s = pd.Series(x)
    out = s.rolling(window=w, center=True, min_periods=max(2, w // 3)).mean()
    return out.bfill().ffill().to_numpy(dtype=float)


def _rolling_var_retrospective(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x)
    out = s.rolling(window=w, center=True, min_periods=max(2, w // 3)).var(ddof=0)
    return out.bfill().ffill().fillna(0.0).to_numpy(dtype=float)


def _contiguous_true_components(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return half-open [i0, i1) components of a 1D boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    out: List[Tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def _persistence_value(t: np.ndarray, i0: int, i1: int, mode: str) -> float:
    if mode == "samples":
        return float(i1 - i0)
    if i1 - i0 <= 1:
        return 0.0
    return float(t[i1 - 1] - t[i0])


def _qualifies_persistence(value: float, pi_min: float, mode: str) -> bool:
    if mode == "samples":
        return int(round(value)) >= int(math.ceil(pi_min))
    return value >= float(pi_min)


# =============================================================================
# Retrospective temporal realization
# =============================================================================

def retrospective_operators(
    f: np.ndarray,
    t: np.ndarray,
    L: Lambda,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrospective temporal D_m and V_m.

    This function is deliberately separate from the live engine. Centered
    windows are allowed here because the complete record is already available.
    The live engine below never calls this function.

    Returns
    -------
    smooth, drift, variation
    """
    L.validate()
    n = len(f)
    if n < 2:
        z = np.zeros(n, dtype=float)
        return f.copy(), z, z

    smooth = _rolling_mean_retrospective(f, int(L.smooth_m))
    w = max(2, min(int(L.m), n))

    if L.D == "diff":
        deriv = np.abs(np.gradient(smooth, t))
    elif L.D == "savgol" and n >= 5:
        sg = min(w, n)
        if sg % 2 == 0:
            sg -= 1
        if sg < 5:
            deriv = np.abs(np.gradient(smooth, t))
        else:
            dt = float(np.median(np.diff(t)))
            poly = min(int(L.sg_polyorder), sg - 2)
            deriv = np.abs(
                savgol_filter(
                    smooth,
                    window_length=sg,
                    polyorder=max(1, poly),
                    deriv=1,
                    delta=dt,
                    mode="interp",
                )
            )
    else:  # local slope / gradient, then local magnitude aggregation
        deriv = np.abs(np.gradient(smooth, t))

    drift = (
        pd.Series(deriv)
        .rolling(window=w, center=True, min_periods=max(2, w // 3))
        .mean()
        .bfill()
        .ffill()
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    var_source = f if L.variation_signal == "raw" else smooth
    variation = _rolling_var_retrospective(var_source, w)
    return smooth, drift, variation


def freeze_lens_from_arrays(drift: np.ndarray, variation: np.ndarray, L: Lambda) -> Dict[str, float]:
    """Freeze eps_d and eps_v once from declared values or quantile proposal."""
    L.validate()
    eps_d = float(L.eps_d) if L.eps_d is not None else _safe_quantile(drift, L.q_d)
    eps_v = float(L.eps_v) if L.eps_v is not None else _safe_quantile(variation, L.q_v)
    return {"eps_d": eps_d, "eps_v": eps_v}


def constitute_temporal(
    f: ArrayLike,
    t: Optional[ArrayLike] = None,
    L: Optional[Lambda] = None,
) -> Dict[str, Any]:
    """Retrospective temporal Plateau constitution followed by measurement.

    This is the direct temporal realization of Algorithm 1 A on a complete
    record. Use PDEngine / replay_live for causal operation.
    """
    y = _as_1d(f, name="f")
    tt = _time(len(y), t)
    tt, y = _finite_pair(tt, y)
    n = len(y)

    if L is None:
        L = Lambda(m=max(5, min(21, max(5, n // 8))))
    L.validate()

    if n == 0:
        return {
            "t": tt,
            "f": y,
            "smooth": y,
            "d": np.array([]),
            "v": np.array([]),
            "C": np.array([], dtype=bool),
            "plateau_mask": np.array([], dtype=bool),
            "components": [],
            "P": [],
            "beta0": 0,
            "Psi": np.array([]),
            "O_gated": np.array([]),
            "Lambda": asdict(L),
        }

    smooth, d, v = retrospective_operators(y, tt, L)
    lens = freeze_lens_from_arrays(d, v, L)

    if L.pi_min is None:
        if L.persistence == "samples":
            pi_min = float(max(3, L.m))
        else:
            dt_med = float(np.median(np.diff(tt))) if n > 1 else 1.0
            pi_min = float(max(3.0 * dt_med, L.m * dt_med))
    else:
        pi_min = float(L.pi_min)

    C = (d <= lens["eps_d"]) & (v <= lens["eps_v"])
    raw_components = _contiguous_true_components(C)
    components: List[Dict[str, Any]] = []
    plateaus: List[Dict[str, Any]] = []
    plateau_mask = np.zeros(n, dtype=bool)

    Psi = 1.0 / (v + float(L.rho))
    dt = np.zeros(n, dtype=float)
    if n > 1:
        dt[1:] = np.diff(tt)
        dt[0] = float(np.median(np.diff(tt)))
    elif n == 1:
        dt[0] = 1.0

    for component_id, (i0, i1) in enumerate(raw_components, start=1):
        Pi = _persistence_value(tt, i0, i1, L.persistence)
        comp = {
            "component_id": component_id,
            "i0": int(i0),
            "i1": int(i1),
            "t0": float(tt[i0]),
            "t1": float(tt[i1 - 1]),
            "Pi": float(Pi),
            "n": int(i1 - i0),
            "mean": float(np.mean(y[i0:i1])),
        }
        components.append(comp)
        if _qualifies_persistence(Pi, pi_min, L.persistence):
            plateau_mask[i0:i1] = True
            p = dict(comp)
            p["id"] = len(plateaus) + 1
            p["beta0"] = 1
            p["O"] = float(np.sum(Psi[i0:i1] * dt[i0:i1]))
            p["height"] = p["mean"]
            plateaus.append(p)

    O_gated = np.cumsum(np.where(plateau_mask, Psi, 0.0) * dt)

    frozen = replace(L, eps_d=lens["eps_d"], eps_v=lens["eps_v"], pi_min=pi_min)
    return {
        "t": tt,
        "f": y,
        "smooth": smooth,
        "d": d,
        "v": v,
        "C": C,
        "plateau_mask": plateau_mask,
        "components": components,
        "P": plateaus,
        "beta0": len(plateaus),
        "Psi": Psi,
        "O_gated": O_gated,
        "Lambda": asdict(frozen),
        "mode": "retrospective",
    }


# Backward-compatible names from the earlier script.
constitute = constitute_temporal
algorithm_1a = constitute_temporal


# =============================================================================
# Fully causal live operators
# =============================================================================

class CausalOperatorBank:
    """Causal local operators using trailing samples only."""

    def __init__(self, L: Lambda):
        self.L = L.validate()
        self.t: List[float] = []
        self.raw: List[float] = []
        self.smooth: List[float] = []

    @property
    def n(self) -> int:
        return len(self.raw)

    def _smooth_last(self) -> float:
        w = min(int(self.L.smooth_m), len(self.raw))
        if w <= 1:
            return float(self.raw[-1])
        return float(np.mean(self.raw[-w:]))

    def _window_ready(self) -> bool:
        if not self.L.require_full_window:
            return self.n >= max(3, min(self.L.m, 3))
        return self.n >= max(int(self.L.m), int(self.L.smooth_m), 3)

    def _drift_last(self) -> float:
        m = min(int(self.L.m), self.n)
        ts = np.asarray(self.t[-m:], dtype=float)
        ys = np.asarray(self.smooth[-m:], dtype=float)

        if len(ys) < 2:
            return float("nan")

        if self.L.D == "diff":
            dt = ts[-1] - ts[-2]
            return abs(float((ys[-1] - ys[-2]) / dt)) if dt > 0 else float("nan")

        if self.L.D == "savgol" and len(ys) >= 5:
            sg = len(ys)
            if sg % 2 == 0:
                sg -= 1
            if sg >= 5:
                yy = ys[-sg:]
                tt = ts[-sg:]
                dt = float(np.median(np.diff(tt)))
                poly = min(int(self.L.sg_polyorder), sg - 2)
                try:
                    deriv = savgol_filter(
                        yy,
                        window_length=sg,
                        polyorder=max(1, poly),
                        deriv=1,
                        delta=dt,
                        mode="interp",
                    )
                    return abs(float(deriv[-1]))
                except ValueError:
                    pass

        x = ts - ts.mean()
        den = float(np.dot(x, x))
        if den <= 0:
            return 0.0
        slope = float(np.dot(x, ys - ys.mean()) / den)
        return abs(slope)

    def _variation_last(self) -> float:
        m = min(int(self.L.m), self.n)
        if self.L.variation_signal == "raw":
            x = np.asarray(self.raw[-m:], dtype=float)
        else:
            x = np.asarray(self.smooth[-m:], dtype=float)
        if len(x) < 2:
            return float("nan")
        return float(np.var(x, ddof=0))

    def feed(self, t_k: float, f_k: float) -> Dict[str, Any]:
        t_k = float(t_k)
        f_k = float(f_k)
        if self.t and t_k <= self.t[-1]:
            raise ValueError("t must be strictly increasing")
        if not (np.isfinite(t_k) and np.isfinite(f_k)):
            raise ValueError("live samples must be finite")

        self.t.append(t_k)
        self.raw.append(f_k)
        self.smooth.append(self._smooth_last())

        if not self._window_ready():
            return {
                "ready": False,
                "t": t_k,
                "f": f_k,
                "smooth": self.smooth[-1],
                "d": None,
                "v": None,
            }

        d = self._drift_last()
        v = self._variation_last()
        return {
            "ready": bool(np.isfinite(d) and np.isfinite(v)),
            "t": t_k,
            "f": f_k,
            "smooth": self.smooth[-1],
            "d": float(d),
            "v": float(v),
        }


# =============================================================================
# Live / real-time state machine
# =============================================================================

class PDEngine:
    """Real-time Plateau Dynamics engine.

    State sequence
    --------------
    idle -> candidate -> confirmed -> exit -> idle

    Important semantics
    -------------------
    - Candidate status is not Plateau identity.
    - Plateau identity becomes operationally available only when Pi >= pi_min.
    - O_live is exactly Plateau-gated: no accumulation occurs before
      confirmation and no candidate history is back-filled.
    - After exit, current membership becomes false while O_live is retained.
    - Thresholds never change after they are frozen.
    """

    def __init__(self, L: Optional[Lambda] = None):
        self.L = (L or Lambda(m=9)).validate()
        self.ops = CausalOperatorBank(self.L)

        self.eps_d: Optional[float] = self.L.eps_d
        self.eps_v: Optional[float] = self.L.eps_v
        self.pi_min: Optional[float] = self.L.pi_min
        self.lens_frozen: bool = self.eps_d is not None and self.eps_v is not None

        self._cal_d: List[float] = []
        self._cal_v: List[float] = []
        self.calibration_points = (
            int(self.L.calibration_points)
            if self.L.calibration_points is not None
            else max(32, 5 * int(self.L.m))
        )

        self.state: str = "idle"
        self.candidate_i0: Optional[int] = None
        self.candidate_t0: Optional[float] = None
        self.candidate_count: int = 0
        self.confirm_i: Optional[int] = None
        self.confirm_t: Optional[float] = None
        self.current_plateau_id: Optional[int] = None
        self.next_plateau_id: int = 1
        self.O_live: float = 0.0
        self.ever_confirmed: bool = False

        self.catalogue: List[Dict[str, Any]] = []
        self.trace: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

        self._last_t: Optional[float] = None
        self._last_state: str = "idle"
        self._last_qualifies: bool = False

    def freeze_lens(self, *, eps_d: Optional[float] = None, eps_v: Optional[float] = None) -> Dict[str, float]:
        """Freeze the live threshold lens exactly once."""
        if self.lens_frozen:
            return {"eps_d": float(self.eps_d), "eps_v": float(self.eps_v)}

        if eps_d is None:
            if self.eps_d is not None:
                eps_d = float(self.eps_d)
            else:
                eps_d = _safe_quantile(np.asarray(self._cal_d), self.L.q_d)
        if eps_v is None:
            if self.eps_v is not None:
                eps_v = float(self.eps_v)
            else:
                eps_v = _safe_quantile(np.asarray(self._cal_v), self.L.q_v)

        self.eps_d = float(eps_d)
        self.eps_v = float(eps_v)
        self.lens_frozen = True
        return {"eps_d": self.eps_d, "eps_v": self.eps_v}

    def frozen_contract(self) -> Dict[str, Any]:
        return {
            **asdict(self.L),
            "eps_d": self.eps_d,
            "eps_v": self.eps_v,
            "pi_min": self._effective_pi_min(),
            "lens_frozen": self.lens_frozen,
            "calibration_points": self.calibration_points,
        }

    def _effective_pi_min(self) -> float:
        if self.pi_min is not None:
            return float(self.pi_min)
        if self.L.persistence == "samples":
            return float(max(3, self.L.m))
        if len(self.ops.t) >= 2:
            dt = float(np.median(np.diff(np.asarray(self.ops.t, dtype=float))))
            return float(max(3.0 * dt, self.L.m * dt))
        return float(self.L.m)

    def _candidate_Pi(self, t_now: float) -> float:
        if self.candidate_t0 is None:
            return 0.0
        if self.L.persistence == "samples":
            return float(self.candidate_count)
        return float(t_now - self.candidate_t0)

    def _candidate_qualifies(self, t_now: float) -> bool:
        return _qualifies_persistence(
            self._candidate_Pi(t_now),
            self._effective_pi_min(),
            self.L.persistence,
        )

    def _emit_event(self, snap: Dict[str, Any]) -> None:
        if snap["event"] not in {"warmup", "calibrating", "idle", "candidate", "confirmed"}:
            self.events.append(dict(snap))

    def feed(self, t_k: float, f_k: float) -> Dict[str, Any]:
        op = self.ops.feed(t_k, f_k)
        idx = self.ops.n - 1
        t_now = float(op["t"])
        dt = 0.0 if self._last_t is None else float(t_now - self._last_t)
        self._last_t = t_now

        if not op["ready"]:
            snap = self._snapshot(
                op,
                idx=idx,
                event="warmup",
                qualifies=False,
                dt=dt,
                delta_O=0.0,
            )
            self.trace.append(snap)
            return snap

        d = float(op["d"])
        v = float(op["v"])

        if not self.lens_frozen:
            if self.L.eps_d is None:
                self._cal_d.append(d)
            if self.L.eps_v is None:
                self._cal_v.append(v)

            enough_d = self.L.eps_d is not None or len(self._cal_d) >= self.calibration_points
            enough_v = self.L.eps_v is not None or len(self._cal_v) >= self.calibration_points
            if enough_d and enough_v:
                self.freeze_lens()
                event = "lens_frozen"
            else:
                event = "calibrating"

            snap = self._snapshot(
                op,
                idx=idx,
                event=event,
                qualifies=False,
                dt=dt,
                delta_O=0.0,
            )
            self.trace.append(snap)
            self._emit_event(snap)
            return snap

        assert self.eps_d is not None and self.eps_v is not None
        qualifies = bool(d <= self.eps_d and v <= self.eps_v)
        psi = 1.0 / (v + float(self.L.rho))

        delta_O = 0.0
        if self._last_state == "confirmed" and qualifies and dt > 0:
            delta_O = float(psi * dt)
            self.O_live += delta_O

        event = "idle"

        if self.state == "idle":
            if qualifies:
                self.state = "candidate"
                self.candidate_i0 = idx
                self.candidate_t0 = t_now
                self.candidate_count = 1
                event = "candidate_on"
            else:
                event = "idle"

        elif self.state == "candidate":
            if qualifies:
                self.candidate_count += 1
                if self._candidate_qualifies(t_now):
                    self.state = "confirmed"
                    self.confirm_i = idx
                    self.confirm_t = t_now
                    self.current_plateau_id = self.next_plateau_id
                    self.next_plateau_id += 1
                    self.ever_confirmed = True
                    event = "confirm"
                else:
                    event = "candidate"
            else:
                event = "candidate_reject"
                self.state = "idle"
                self.candidate_i0 = None
                self.candidate_t0 = None
                self.candidate_count = 0

        elif self.state == "confirmed":
            if qualifies:
                event = "confirmed"
            else:
                event = "exit"
                self._close_current_plateau(exit_i=idx, exit_t=t_now)
                self.state = "idle"
                self.candidate_i0 = None
                self.candidate_t0 = None
                self.candidate_count = 0
                self.confirm_i = None
                self.confirm_t = None
                self.current_plateau_id = None

        snap = self._snapshot(
            op,
            idx=idx,
            event=event,
            qualifies=qualifies,
            dt=dt,
            delta_O=delta_O,
        )
        self.trace.append(snap)
        self._emit_event(snap)
        self._last_state = self.state
        self._last_qualifies = qualifies
        return snap

    def _close_current_plateau(self, *, exit_i: int, exit_t: float) -> None:
        if self.current_plateau_id is None or self.candidate_i0 is None or self.candidate_t0 is None:
            return
        confirm_i = self.confirm_i if self.confirm_i is not None else exit_i
        confirm_t = self.confirm_t if self.confirm_t is not None else exit_t
        previous_t = float(self.ops.t[exit_i - 1]) if exit_i > 0 else float(exit_t)
        rec = {
            "id": int(self.current_plateau_id),
            "candidate_i0": int(self.candidate_i0),
            "candidate_t0": float(self.candidate_t0),
            "confirm_i": int(confirm_i),
            "confirm_t": float(confirm_t),
            "exit_i": int(exit_i),
            "exit_t": float(exit_t),
            "last_qualifying_t": previous_t,
            "candidate_Pi_at_confirmation": float(
                (confirm_i - self.candidate_i0 + 1)
                if self.L.persistence == "samples"
                else (confirm_t - self.candidate_t0)
            ),
            "O_at_exit": float(self.O_live),
            "beta0": 1,
            "status": "closed",
        }
        self.catalogue.append(rec)

    def _snapshot(
        self,
        op: Dict[str, Any],
        *,
        idx: int,
        event: str,
        qualifies: bool,
        dt: float,
        delta_O: float,
    ) -> Dict[str, Any]:
        return {
            "i": int(idx),
            "t": float(op["t"]),
            "f": float(op["f"]),
            "smooth": float(op["smooth"]),
            "ready": bool(op["ready"]),
            "lens_frozen": bool(self.lens_frozen),
            "eps_d": self.eps_d,
            "eps_v": self.eps_v,
            "pi_min": self._effective_pi_min(),
            "d": None if op["d"] is None else float(op["d"]),
            "v": None if op["v"] is None else float(op["v"]),
            "C": bool(qualifies),
            "state": self.state,
            "event": event,
            "candidate_count": int(self.candidate_count),
            "candidate_Pi": float(self._candidate_Pi(float(op["t"]))),
            "confirmed_membership": bool(self.state == "confirmed" and qualifies),
            "current_plateau_id": self.current_plateau_id,
            "dt": float(dt),
            "delta_O": float(delta_O),
            "O_live": float(self.O_live),
            "ever_confirmed": bool(self.ever_confirmed),
        }

    def snapshot(self) -> Optional[Dict[str, Any]]:
        return dict(self.trace[-1]) if self.trace else None

    def finalize(self) -> Dict[str, Any]:
        open_plateau = None
        if self.state == "confirmed" and self.current_plateau_id is not None:
            open_plateau = {
                "id": int(self.current_plateau_id),
                "candidate_i0": self.candidate_i0,
                "candidate_t0": self.candidate_t0,
                "confirm_i": self.confirm_i,
                "confirm_t": self.confirm_t,
                "O_current": float(self.O_live),
                "status": "open",
            }
        return {
            "state": self.state,
            "O_live": float(self.O_live),
            "beta0_closed": len(self.catalogue),
            "open_plateau": open_plateau,
            "Lambda": self.frozen_contract(),
        }


# =============================================================================
# Live replay / file streaming
# =============================================================================

def replay_live(
    f: ArrayLike,
    t: Optional[ArrayLike] = None,
    L: Optional[Lambda] = None,
) -> Dict[str, Any]:
    """Replay a complete record through exactly the causal live engine."""
    y = _as_1d(f, name="f")
    tt = _time(len(y), t)
    tt, y = _finite_pair(tt, y)
    eng = PDEngine(L)
    for ti, yi in zip(tt, y):
        eng.feed(float(ti), float(yi))
    return {
        "engine": eng,
        "trace": pd.DataFrame(eng.trace),
        "events": pd.DataFrame(eng.events),
        "P": pd.DataFrame(eng.catalogue),
        "final": eng.finalize(),
        "Lambda": eng.frozen_contract(),
        "mode": "causal_live_replay",
    }


def _load_numeric_file(
    path: Union[str, Path],
    *,
    time_column: Optional[str] = None,
    signal_column: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    src = Path(path)
    suffix = src.suffix.lower()

    if suffix == ".npy":
        y = np.asarray(np.load(src), dtype=float).ravel()
        t = np.arange(len(y), dtype=float)
    elif suffix == ".npz":
        z = np.load(src)
        keys = list(z.keys())
        if signal_column is not None and signal_column in z:
            y = np.asarray(z[signal_column], dtype=float).ravel()
        else:
            y = np.asarray(z[keys[0]], dtype=float).ravel()
        if time_column is not None and time_column in z:
            t = np.asarray(z[time_column], dtype=float).ravel()
        else:
            t = np.arange(len(y), dtype=float)
    else:
        if suffix in {".tsv", ".txt"}:
            df = pd.read_csv(src, sep=None, engine="python")
        else:
            df = pd.read_csv(src)

        if signal_column is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if time_column is not None:
                numeric = [c for c in numeric if c != time_column]
            if not numeric:
                raise ValueError("no numeric signal column found; use --signal-column")
            signal_column = str(numeric[0])

        y = pd.to_numeric(df[signal_column], errors="coerce").to_numpy(dtype=float)
        if time_column is not None:
            t = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float)
        else:
            t = np.arange(len(y), dtype=float)

    if len(t) != len(y):
        raise ValueError("time and signal arrays have different lengths")
    return _finite_pair(np.asarray(t, dtype=float), np.asarray(y, dtype=float))


def stream_file(
    path: Union[str, Path],
    *,
    time_column: Optional[str] = None,
    signal_column: Optional[str] = None,
    L: Optional[Lambda] = None,
) -> Tuple[PDEngine, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read a real file and feed one observation at a time through PD."""
    t, y = _load_numeric_file(path, time_column=time_column, signal_column=signal_column)
    eng = PDEngine(L)
    for ti, yi in zip(t, y):
        eng.feed(float(ti), float(yi))
    return eng, eng.events, eng.trace


# =============================================================================
# Spatial realization
# =============================================================================

def _neighbors(adjacency: Adjacency, p: int) -> Sequence[int]:
    if callable(adjacency):
        return adjacency(int(p))
    if isinstance(adjacency, Mapping):
        return adjacency.get(int(p), [])
    return adjacency[int(p)]


def _graph_components(mask: np.ndarray, adjacency: Adjacency) -> List[np.ndarray]:
    active = set(np.flatnonzero(mask).tolist())
    seen: set[int] = set()
    comps: List[np.ndarray] = []

    for p in sorted(active):
        if p in seen:
            continue
        stack = [p]
        seen.add(p)
        cur: List[int] = []
        while stack:
            q = stack.pop()
            cur.append(q)
            for r in _neighbors(adjacency, q):
                r = int(r)
                if r in active and r not in seen:
                    seen.add(r)
                    stack.append(r)
        comps.append(np.asarray(sorted(cur), dtype=int))
    return comps


def spatial_operators(
    values: ArrayLike,
    adjacency: Adjacency,
    *,
    analysis_valid: Optional[np.ndarray] = None,
    source_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatial D_m and V_m under declared adjacency.

    d(p) = mean_n |f(n)-f(p)|
    v(p) = population std of {f(p)} U {f(n)}

    `analysis_valid` controls which centres may become candidates.
    `source_valid` controls which neighbours may contribute to local statistics.
    This matches the manuscript distinction between analysis-valid centres and
    source-valid neighbours at mask boundaries.
    """
    x = _as_1d(values, name="values")
    n = len(x)

    finite = np.isfinite(x)
    if source_valid is None:
        source_valid = finite.copy()
    else:
        source_valid = np.asarray(source_valid, dtype=bool).ravel() & finite
        if len(source_valid) != n:
            raise ValueError("source_valid has wrong length")

    if analysis_valid is None:
        analysis_valid = source_valid.copy()
    else:
        analysis_valid = np.asarray(analysis_valid, dtype=bool).ravel() & finite
        if len(analysis_valid) != n:
            raise ValueError("analysis_valid has wrong length")

    d = np.full(n, np.nan, dtype=float)
    v = np.full(n, np.nan, dtype=float)

    for p in np.flatnonzero(analysis_valid):
        nb = [int(q) for q in _neighbors(adjacency, int(p)) if 0 <= int(q) < n and source_valid[int(q)]]
        if not nb:
            continue
        vals = x[np.asarray(nb, dtype=int)]
        d[p] = float(np.mean(np.abs(vals - x[p])))
        v[p] = float(np.std(np.r_[x[p], vals], ddof=0))

    local_valid = analysis_valid & np.isfinite(d) & np.isfinite(v)
    return d, v, local_valid


def constitute_spatial(
    values: ArrayLike,
    adjacency: Adjacency,
    *,
    eps_d: Optional[float] = None,
    eps_v: Optional[float] = None,
    q_d: float = 0.25,
    q_v: float = 0.25,
    n_min: int = 8,
    rho: float = 1e-9,
    analysis_valid: Optional[np.ndarray] = None,
    source_valid: Optional[np.ndarray] = None,
    element_weight: Optional[np.ndarray] = None,
    order_u: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Spatial Plateau constitution followed by post-identity measurement."""
    x = _as_1d(values, name="values")
    n = len(x)
    if n_min < 1:
        raise ValueError("n_min must be >= 1")

    d, v, local_valid = spatial_operators(
        x,
        adjacency,
        analysis_valid=analysis_valid,
        source_valid=source_valid,
    )

    if not np.any(local_valid):
        raise ValueError("no valid spatial centres with local statistics")

    ed = float(eps_d) if eps_d is not None else _safe_quantile(d[local_valid], q_d)
    ev = float(eps_v) if eps_v is not None else _safe_quantile(v[local_valid], q_v)

    C = np.zeros(n, dtype=bool)
    C[local_valid] = (d[local_valid] <= ed) & (v[local_valid] <= ev)
    raw = _graph_components(C, adjacency)

    if element_weight is None:
        omega = np.ones(n, dtype=float)
    else:
        omega = np.asarray(element_weight, dtype=float).ravel()
        if len(omega) != n:
            raise ValueError("element_weight has wrong length")

    if order_u is None:
        u = v.copy()
        order_u_source = "local_variation_v"
    else:
        u = np.asarray(order_u, dtype=float).ravel()
        if len(u) != n:
            raise ValueError("order_u has wrong length")
        order_u_source = "caller_supplied_u"

    mask = np.zeros(n, dtype=bool)
    P: List[Dict[str, Any]] = []

    for comp in raw:
        if len(comp) < int(n_min):
            continue
        mask[comp] = True
        uu = u[comp]
        good = np.isfinite(uu)
        if not np.all(good):
            fill = float(np.nanmedian(uu)) if np.any(good) else 0.0
            uu = np.where(good, uu, fill)
        O = float(np.sum(omega[comp] / (uu + float(rho))))
        P.append(
            {
                "id": len(P) + 1,
                "pixels": comp,
                "n": int(len(comp)),
                "mean": float(np.mean(x[comp])),
                "O": O,
                "beta0": 1,
            }
        )

    return {
        "values": x,
        "d": d,
        "v": v,
        "local_valid": local_valid,
        "C": C,
        "components": raw,
        "P": P,
        "mask": mask,
        "beta0": len(P),
        "eps_d": ed,
        "eps_v": ev,
        "q_d": q_d,
        "q_v": q_v,
        "n_min": int(n_min),
        "rho": float(rho),
        "order_u_source": order_u_source,
        "mode": "spatial",
    }


# =============================================================================
# HEALPix wrapper using official healpy neighbours
# =============================================================================

def nside2npix(nside: int) -> int:
    if int(nside) < 1 or (int(nside) & (int(nside) - 1)) != 0:
        raise ValueError("nside must be a positive power of two")
    return 12 * int(nside) * int(nside)


def healpix_first_ring_adjacency(nside: int, *, nest: bool = True) -> Callable[[int], Sequence[int]]:
    """Return official HEALPix first-ring adjacency via healpy.

    A hand-written face-wrap approximation is intentionally not used here.
    Scientific HEALPix runs should use the authoritative topology.
    """
    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError(
            "HEALPix mode requires healpy. Install with `pip install healpy`."
        ) from exc

    ns = int(nside)

    def neighbours(p: int) -> Sequence[int]:
        nb = hp.get_all_neighbours(ns, int(p), nest=bool(nest))
        return [int(q) for q in nb if int(q) >= 0]

    return neighbours


def algorithm_1b(
    T: ArrayLike,
    *,
    nside: int,
    q: float = 0.25,
    n_min: int = 8,
    analysis_valid: Optional[np.ndarray] = None,
    source_valid: Optional[np.ndarray] = None,
    nest: bool = True,
    eps_d: Optional[float] = None,
    eps_v: Optional[float] = None,
    order_u: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Manuscript Algorithm 1 B — HEALPix spatial/CMB realization."""
    x = _as_1d(T, name="T")
    expected = nside2npix(nside)
    if len(x) != expected:
        raise ValueError(f"T length {len(x)} != 12*nside^2 = {expected}")

    adjacency = healpix_first_ring_adjacency(nside, nest=nest)
    out = constitute_spatial(
        x,
        adjacency,
        eps_d=eps_d,
        eps_v=eps_v,
        q_d=q,
        q_v=q,
        n_min=n_min,
        analysis_valid=analysis_valid,
        source_valid=source_valid,
        order_u=order_u,
    )
    out.update({"nside": int(nside), "nest": bool(nest), "q": float(q)})
    return out


# =============================================================================
# Cross-scale lineage helper
# =============================================================================

def build_lineages(
    parent_result: Dict[str, Any],
    child_result: Dict[str, Any],
    parent_of_child: Sequence[int],
) -> Dict[str, Any]:
    """Link constituted child formations to constituted parent objects.

    `parent_of_child[c]` gives the parent-grid element containing child c.
    The helper does not alter Plateau identity; it only relates already
    constituted objects across scales.
    """
    poc = np.asarray(parent_of_child, dtype=int).ravel()
    parent_P = parent_result.get("P", [])
    child_P = child_result.get("P", [])

    parent_pixel_to_id: Dict[int, int] = {}
    for p in parent_P:
        members = p.get("pixels")
        if members is None:
            continue
        for pix in np.asarray(members, dtype=int).ravel():
            parent_pixel_to_id[int(pix)] = int(p["id"])

    lineages: List[Dict[str, Any]] = []
    by_parent: Dict[int, List[int]] = {}

    for child in child_P:
        cpix = np.asarray(child.get("pixels", []), dtype=int).ravel()
        cpix = cpix[(cpix >= 0) & (cpix < len(poc))]
        mapped = poc[cpix]
        parent_ids = sorted({parent_pixel_to_id[int(pp)] for pp in mapped if int(pp) in parent_pixel_to_id})
        for pid in parent_ids:
            by_parent.setdefault(pid, []).append(int(child["id"]))

    for pid in sorted(by_parent):
        lineages.append(
            {
                "parent_id": int(pid),
                "child_ids": sorted(set(by_parent[pid])),
                "n_child_formations": int(len(set(by_parent[pid]))),
                "qualifies": True,
            }
        )

    return {
        "lineages": lineages,
        "K": len(lineages),
        "child_formations": sum(x["n_child_formations"] for x in lineages),
    }


# =============================================================================
# Export helpers
# =============================================================================

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def export_temporal_result(result: Dict[str, Any], prefix: Union[str, Path]) -> Dict[str, str]:
    """Export retrospective temporal trace, Plateau catalogue, and contract."""
    p = Path(prefix)
    p.parent.mkdir(parents=True, exist_ok=True)

    trace_path = str(p.with_name(p.name + "_trace.csv"))
    plateaus_path = str(p.with_name(p.name + "_plateaus.csv"))
    contract_path = str(p.with_name(p.name + "_contract.json"))

    pd.DataFrame(
        {
            "t": result["t"],
            "f": result["f"],
            "smooth": result["smooth"],
            "d": result["d"],
            "v": result["v"],
            "C": result["C"],
            "plateau_mask": result["plateau_mask"],
            "Psi": result["Psi"],
            "O_gated": result["O_gated"],
        }
    ).to_csv(trace_path, index=False)
    pd.DataFrame(result["P"]).to_csv(plateaus_path, index=False)
    Path(contract_path).write_text(json.dumps(_jsonable(result["Lambda"]), indent=2), encoding="utf-8")
    return {"trace": trace_path, "plateaus": plateaus_path, "contract": contract_path}


def export_live_result(engine: PDEngine, prefix: Union[str, Path]) -> Dict[str, str]:
    p = Path(prefix)
    p.parent.mkdir(parents=True, exist_ok=True)

    trace_path = str(p.with_name(p.name + "_live_trace.csv"))
    event_path = str(p.with_name(p.name + "_events.csv"))
    plateau_path = str(p.with_name(p.name + "_live_plateaus.csv"))
    contract_path = str(p.with_name(p.name + "_live_contract.json"))

    pd.DataFrame(engine.trace).to_csv(trace_path, index=False)
    pd.DataFrame(engine.events).to_csv(event_path, index=False)
    pd.DataFrame(engine.catalogue).to_csv(plateau_path, index=False)
    Path(contract_path).write_text(
        json.dumps(_jsonable(engine.frozen_contract()), indent=2), encoding="utf-8"
    )
    return {
        "trace": trace_path,
        "events": event_path,
        "plateaus": plateau_path,
        "contract": contract_path,
    }


def export_spatial_result(result: Dict[str, Any], prefix: Union[str, Path]) -> Dict[str, str]:
    p = Path(prefix)
    p.parent.mkdir(parents=True, exist_ok=True)

    pixel_path = str(p.with_name(p.name + "_pixels.csv"))
    plateau_path = str(p.with_name(p.name + "_formations.json"))

    pd.DataFrame(
        {
            "value": result["values"],
            "d": result["d"],
            "v": result["v"],
            "local_valid": result["local_valid"],
            "C": result["C"],
            "plateau_mask": result["mask"],
        }
    ).to_csv(pixel_path, index_label="element")
    Path(plateau_path).write_text(json.dumps(_jsonable(result["P"]), indent=2), encoding="utf-8")
    return {"pixels": pixel_path, "formations": plateau_path}


# =============================================================================
# CLI
# =============================================================================

def _lambda_from_args(a: argparse.Namespace) -> Lambda:
    return Lambda(
        m=int(a.window),
        D=str(a.drift),
        V="rolling_var",
        eps_d=a.eps_d,
        eps_v=a.eps_v,
        pi_min=a.pi_min,
        q_d=float(a.q_d),
        q_v=float(a.q_v),
        calibration_points=a.calibration_points,
        persistence=str(a.persistence),
        smooth_m=int(a.smooth_window),
        variation_signal=str(a.variation_signal),
        sg_polyorder=int(a.sg_polyorder),
        require_full_window=not bool(a.allow_partial_window),
        rho=float(a.rho),
    ).validate()


def _add_temporal_contract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--window", type=int, default=9, help="local detector window m")
    p.add_argument("--smooth-window", type=int, default=1, help="trailing/centered smoothing window")
    p.add_argument("--drift", choices=["slope", "diff", "savgol"], default="slope")
    p.add_argument("--variation-signal", choices=["raw", "smoothed"], default="raw")
    p.add_argument("--sg-polyorder", type=int, default=2)
    p.add_argument("--eps-d", type=float)
    p.add_argument("--eps-v", type=float)
    p.add_argument("--q-d", type=float, default=0.40)
    p.add_argument("--q-v", type=float, default=0.40)
    p.add_argument("--pi-min", type=float)
    p.add_argument("--persistence", choices=["time", "samples"], default="time")
    p.add_argument("--calibration-points", type=int)
    p.add_argument("--rho", type=float, default=1e-9)
    p.add_argument("--allow-partial-window", action="store_true")


def _print_live_events(engine: PDEngine) -> None:
    for s in engine.events:
        print(
            f"t={s['t']:.8g}  event={s['event']:16s}  "
            f"state={s['state']:10s}  d={s['d']}  v={s['v']}  "
            f"O={s['O_live']:.8g}"
        )


def cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plateau Dynamics standalone temporal/live/spatial engine"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ret = sub.add_parser("temporal", help="retrospective temporal constitution")
    p_ret.add_argument("path")
    p_ret.add_argument("--time-column")
    p_ret.add_argument("--signal-column")
    p_ret.add_argument("--export-prefix")
    _add_temporal_contract_args(p_ret)

    p_live = sub.add_parser("live", help="causal sample-by-sample replay of a real file")
    p_live.add_argument("path")
    p_live.add_argument("--time-column")
    p_live.add_argument("--signal-column")
    p_live.add_argument("--export-prefix")
    _add_temporal_contract_args(p_live)

    p_hp = sub.add_parser("healpix", help="HEALPix spatial Plateau constitution from .npy")
    p_hp.add_argument("path", help=".npy temperature/field array")
    p_hp.add_argument("--nside", type=int, required=True)
    p_hp.add_argument("--q", type=float, default=0.25)
    p_hp.add_argument("--n-min", type=int, default=8)
    p_hp.add_argument("--eps-d", type=float)
    p_hp.add_argument("--eps-v", type=float)
    p_hp.add_argument("--ring", action="store_true", help="input is RING instead of NESTED")
    p_hp.add_argument("--export-prefix")

    a = parser.parse_args(argv)

    if a.command == "temporal":
        t, y = _load_numeric_file(a.path, time_column=a.time_column, signal_column=a.signal_column)
        L = _lambda_from_args(a)
        out = constitute_temporal(y, t, L)
        print(
            f"samples={len(y)}  beta0={out['beta0']}  "
            f"eps_d={out['Lambda']['eps_d']:.8g}  "
            f"eps_v={out['Lambda']['eps_v']:.8g}  "
            f"pi_min={out['Lambda']['pi_min']}"
        )
        if a.export_prefix:
            paths = export_temporal_result(out, a.export_prefix)
            print(json.dumps(paths, indent=2))
        return 0

    if a.command == "live":
        L = _lambda_from_args(a)
        eng, _, _ = stream_file(
            a.path,
            time_column=a.time_column,
            signal_column=a.signal_column,
            L=L,
        )
        print(json.dumps(_jsonable(eng.finalize()), indent=2))
        _print_live_events(eng)
        if a.export_prefix:
            paths = export_live_result(eng, a.export_prefix)
            print(json.dumps(paths, indent=2))
        return 0

    if a.command == "healpix":
        x = np.asarray(np.load(a.path), dtype=float).ravel()
        out = algorithm_1b(
            x,
            nside=a.nside,
            q=a.q,
            n_min=a.n_min,
            nest=not a.ring,
            eps_d=a.eps_d,
            eps_v=a.eps_v,
        )
        print(
            f"nside={a.nside}  beta0={out['beta0']}  "
            f"candidate_pixels={int(np.sum(out['C']))}  "
            f"plateau_pixels={int(np.sum(out['mask']))}  "
            f"eps_d={out['eps_d']:.8g}  eps_v={out['eps_v']:.8g}"
        )
        if a.export_prefix:
            paths = export_spatial_result(out, a.export_prefix)
            print(json.dumps(paths, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
