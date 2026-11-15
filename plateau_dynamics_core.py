#!/usr/bin/env python3
"""Standalone causal/live Plateau Dynamics reference engine.

This repository is independent from the UPM engine. It implements causal local
operators, frozen threshold calibration, persistence-confirmed plateaus, live
Order, replay, CSV/NPY streaming, and graph-based spatial constitution.

Ruben Kabongo, 2026
ORCID: 0009-0005-5573-0857
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Lambda:
    """Frozen measurement contract for the causal engine."""
    m: int
    pi_min: float
    eps_d: Optional[float] = None
    eps_v: Optional[float] = None
    q_d: float = 0.40
    q_v: float = 0.40
    calibration_points: int = 256
    D: str = "poly2_right"
    persistence: str = "duration"
    rho: float = 1e-9

    def __post_init__(self) -> None:
        if self.m < 3 or self.pi_min <= 0:
            raise ValueError("m must be >= 3 and pi_min must be > 0")
        if (self.eps_d is None) != (self.eps_v is None):
            raise ValueError("eps_d and eps_v must be supplied together")
        if self.eps_d is not None and (self.eps_d < 0 or self.eps_v < 0):
            raise ValueError("eps_d and eps_v must be non-negative")
        if not (0 < self.q_d < 1 and 0 < self.q_v < 1):
            raise ValueError("quantiles must be between 0 and 1")
        if self.calibration_points < 8:
            raise ValueError("calibration_points must be >= 8")
        if self.D not in {"poly2_right", "linear_right", "finite_difference"}:
            raise ValueError("unsupported drift estimator")
        if self.persistence not in {"duration", "samples"}:
            raise ValueError("persistence must be 'duration' or 'samples'")
        if self.rho <= 0:
            raise ValueError("rho must be > 0")


def _derivative(t: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "finite_difference":
        return float((y[-1] - y[-2]) / (t[-1] - t[-2]))
    x = t - t[-1]
    degree = 1 if method == "linear_right" else 2
    coef = np.polyfit(x, y, degree)
    return float(coef[-2])


def causal_operators(t_window: Sequence[float], f_window: Sequence[float], L: Lambda) -> Tuple[float, float]:
    """Return causal absolute right-edge drift and population variance."""
    t = np.asarray(t_window, float)
    f = np.asarray(f_window, float)
    if len(t) != len(f) or len(t) < L.m:
        raise ValueError("a complete equal-length trailing window is required")
    if np.any(np.diff(t) <= 0) or not np.all(np.isfinite(f)):
        raise ValueError("timestamps must increase and signal values be finite")
    return abs(_derivative(t[-L.m:], f[-L.m:], L.D)), float(np.var(f[-L.m:]))


class PDEngine:
    """Causal state machine. Feed observations with ``feed(t, f)``."""

    def __init__(self, L: Lambda):
        self.L = L
        self.t: List[float] = []; self.f: List[float] = []
        self.d: List[float] = []; self.v: List[float] = []
        self.C: List[bool] = []; self.confirmed_mask: List[bool] = []
        self.states: List[str] = []; self.events: List[str] = []
        self.O_live_history: List[float] = []
        self._cal_d: List[float] = []; self._cal_v: List[float] = []
        self._eps_d = L.eps_d; self._eps_v = L.eps_v
        self._candidate_i: Optional[int] = None; self._candidate_t: Optional[float] = None
        self._count = 0; self._confirmed = False; self._current = None
        self._next_id = 1; self.plateaus: List[Dict] = []; self.O_live = 0.0

    @property
    def lens_frozen(self): return self._eps_d is not None
    @property
    def eps_d(self): return self._eps_d
    @property
    def eps_v(self): return self._eps_v
    @property
    def confirmed(self): return self._confirmed
    @property
    def frozen_lambda(self):
        if not self.lens_frozen: return None
        out = asdict(self.L); out.update(eps_d=float(self._eps_d), eps_v=float(self._eps_v)); return out

    def _pi(self, now):
        if self._candidate_t is None: return 0.0
        return float(self._count if self.L.persistence == "samples" else now - self._candidate_t)

    def _close(self, last: int):
        p = self._current; p.update(i1=last + 1, t1=self.t[last], n=last-p["i0"]+1,
            Pi=(last-p["i0"]+1 if self.L.persistence == "samples" else self.t[last]-p["t0"]),
            mean=float(np.mean(self.f[p["i0"]:last+1])), open=False)
        self.plateaus.append(dict(p)); self._current = None; self._confirmed = False

    def feed(self, t_k: float, f_k: float) -> Dict:
        t_k, f_k = float(t_k), float(f_k)
        if not np.isfinite(t_k) or not np.isfinite(f_k): raise ValueError("inputs must be finite")
        if self.t and t_k <= self.t[-1]: raise ValueError("timestamps must be strictly increasing")
        previous_confirmed = self._confirmed; previous_t = self.t[-1] if self.t else None
        self.t.append(t_k); self.f.append(f_k); idx = len(self.f)-1
        d = v = np.nan; C = False; state = "warmup"; event = "warmup"
        if len(self.f) >= self.L.m:
            d, v = causal_operators(self.t[-self.L.m:], self.f[-self.L.m:], self.L)
            if not self.lens_frozen:
                self._cal_d.append(d); self._cal_v.append(v)
                if len(self._cal_d) >= self.L.calibration_points:
                    tiny = np.finfo(float).eps
                    self._eps_d = max(float(np.quantile(self._cal_d, self.L.q_d)), tiny)
                    self._eps_v = max(float(np.quantile(self._cal_v, self.L.q_v)), tiny)
                    event, state = "lens_frozen", "search"
                else: event, state = "calibrating", "calibrating"
            else:
                C = bool(d <= self._eps_d and v <= self._eps_v)
                if previous_confirmed and C and previous_t is not None:
                    self.O_live += (1.0/(v+self.L.rho)) * (t_k-previous_t)
                    self._current["O_live"] += (1.0/(v+self.L.rho)) * (t_k-previous_t)
                if C:
                    if self._candidate_i is None:
                        self._candidate_i, self._candidate_t, self._count = idx, t_k, 1; event = "candidate_on"
                    else: self._count += 1; event = "candidate_hold"
                    if not self._confirmed and self._pi(t_k) >= self.L.pi_min:
                        self._confirmed = True
                        self._current = dict(id=self._next_id, i0=self._candidate_i, t0=self._candidate_t,
                            i_confirm=idx, t_confirm=t_k, Pi_at_confirm=self._pi(t_k), O_live=0.0, open=True)
                        self._next_id += 1; event = "plateau_on"
                    elif self._confirmed: event = "plateau_hold"
                    state = "confirmed" if self._confirmed else "candidate"
                else:
                    if self._confirmed: self._close(idx-1); event = "plateau_off"
                    elif self._candidate_i is not None: event = "candidate_reset"
                    else: event = "search"
                    self._candidate_i = self._candidate_t = None; self._count = 0; state = "search"
        self.d.append(float(d)); self.v.append(float(v)); self.C.append(C)
        self.confirmed_mask.append(self._confirmed); self.states.append(state); self.events.append(event)
        self.O_live_history.append(self.O_live)
        return self.snapshot()

    def snapshot(self) -> Dict:
        if not self.t: return {"ready": False, "state": "empty", "event": "empty", "confirmed": False, "O_live": 0.0}
        return {"index": len(self.t)-1, "t": self.t[-1], "f": self.f[-1],
            "ready": len(self.f) >= self.L.m and self.lens_frozen, "lens_frozen": self.lens_frozen,
            "eps_d": self.eps_d, "eps_v": self.eps_v,
            "d": None if not np.isfinite(self.d[-1]) else self.d[-1],
            "v": None if not np.isfinite(self.v[-1]) else self.v[-1], "C": self.C[-1],
            "state": self.states[-1], "event": self.events[-1], "candidate": self._candidate_i is not None,
            "confirmed": self._confirmed, "plateau_id": self._current["id"] if self._current else None,
            "Pi": self._pi(self.t[-1]), "O_live": self.O_live, "beta0_closed": len(self.plateaus)}

    def plateau_catalogue(self, include_open=True) -> List[Dict]:
        out = [dict(p) for p in self.plateaus]
        if include_open and self._current:
            p = dict(self._current); last = len(self.t)-1
            p.update(i1=last+1, t1=self.t[last], n=last-p["i0"]+1,
                Pi=(last-p["i0"]+1 if self.L.persistence == "samples" else self.t[last]-p["t0"]),
                mean=float(np.mean(self.f[p["i0"]:last+1])))
            out.append(p)
        return out

    def result(self) -> Dict:
        n = len(self.t); mask = np.zeros(n, bool)
        for p in self.plateau_catalogue(): mask[p["i0"]:p["i1"]] = True
        tt = np.asarray(self.t); vv = np.asarray(self.v); O = np.zeros(n)
        if n > 1:
            inside = np.zeros(n, bool); inside[1:] = mask[:-1] & mask[1:]
            psi = np.where(np.isfinite(vv), 1/(vv+self.L.rho), 0)
            O = np.cumsum(np.where(inside, psi*np.r_[0, np.diff(tt)], 0))
        return {"t": tt, "f": np.asarray(self.f), "d": np.asarray(self.d), "v": vv,
            "C": np.asarray(self.C), "confirmed_mask": np.asarray(self.confirmed_mask),
            "plateau_mask": mask, "state": list(self.states), "event": list(self.events),
            "O_live": np.asarray(self.O_live_history), "O_retrospective": O,
            "P": self.plateau_catalogue(), "beta0": len(self.plateau_catalogue()), "Lambda": self.frozen_lambda}


def replay(f: Sequence[float], t: Optional[Sequence[float]], L: Lambda) -> Dict:
    y = np.asarray(f, float).ravel(); tt = np.arange(len(y), dtype=float) if t is None else np.asarray(t, float).ravel()
    if len(y) != len(tt): raise ValueError("f and t length mismatch")
    e = PDEngine(L)
    for ti, yi in zip(tt, y): e.feed(ti, yi)
    return e.result()


def stream_file(path: str, *, L: Lambda, time_column=None, signal_column=None, chunksize=50000):
    e = PDEngine(L); src = Path(path); events = []
    if src.suffix.lower() == ".npy":
        y = np.asarray(np.load(src, mmap_mode="r")).reshape(-1); rows = enumerate(y)
        for i, yi in rows:
            if np.isfinite(yi):
                s=e.feed(i, yi)
                if s["event"] in {"lens_frozen", "plateau_on", "plateau_off"}: events.append(dict(s))
        return e, events
    if src.suffix.lower() not in {".csv", ".txt"}: raise ValueError("supported files: .csv, .txt, .npy")
    offset = 0; selected = signal_column
    for chunk in pd.read_csv(src, chunksize=chunksize):
        if selected is None:
            numeric = chunk.select_dtypes(include=[np.number]).columns.tolist()
            if time_column in numeric: numeric.remove(time_column)
            if not numeric: raise ValueError("no numeric signal column found")
            selected = numeric[0]
        y = pd.to_numeric(chunk[selected], errors="coerce").to_numpy(float)
        t = np.arange(offset, offset+len(chunk), dtype=float) if time_column is None else pd.to_numeric(chunk[time_column], errors="coerce").to_numpy(float)
        for ti, yi in zip(t, y):
            if np.isfinite(ti) and np.isfinite(yi):
                s=e.feed(ti, yi)
                if s["event"] in {"lens_frozen", "plateau_on", "plateau_off"}: events.append(dict(s))
        offset += len(chunk)
    return e, events


def constitute_spatial(field, *, neighbors: Callable[[int], Iterable[int]], n_min: int, valid=None, eps_d=None, eps_v=None, q=.25, rho=1e-9):
    x=np.asarray(field,float).ravel(); n=len(x); ok=np.isfinite(x) if valid is None else np.asarray(valid,bool)&np.isfinite(x)
    if len(ok)!=n or n_min<1 or rho<=0: raise ValueError("invalid spatial arguments")
    d=np.full(n,np.nan); v=np.full(n,np.nan)
    for p in np.flatnonzero(ok):
        nb=[int(j) for j in neighbors(int(p)) if 0<=int(j)<n and np.isfinite(x[int(j)])]
        if nb: d[p]=np.mean(np.abs(x[nb]-x[p])); v[p]=np.std(np.r_[x[p],x[nb]])
    usable=ok&np.isfinite(d)&np.isfinite(v)
    if not usable.any(): raise ValueError("no usable spatial centers")
    if (eps_d is None)!=(eps_v is None): raise ValueError("eps_d and eps_v must be supplied together")
    eps_d=float(np.quantile(d[usable],q)) if eps_d is None else float(eps_d); eps_v=float(np.quantile(v[usable],q)) if eps_v is None else float(eps_v)
    C=usable&(d<=eps_d)&(v<=eps_v); active=set(np.flatnonzero(C)); seen=set(); P=[]
    for start in sorted(active):
        if start in seen: continue
        stack=[start]; seen.add(start); comp=[]
        while stack:
            p=stack.pop(); comp.append(p)
            for q0 in neighbors(p):
                q0=int(q0)
                if q0 in active and q0 not in seen: seen.add(q0); stack.append(q0)
        if len(comp)>=n_min: P.append({"members":np.asarray(comp),"n":len(comp),"mean":float(np.mean(x[comp])),"O":float(np.sum(1/(v[comp]+rho))),"beta0":1})
    mask=np.zeros(n,bool)
    for p in P: mask[p["members"]]=True
    return {"field":x,"d":d,"v":v,"C":C,"mask":mask,"P":P,"beta0":len(P),"eps_d":eps_d,"eps_v":eps_v,"n_min":n_min}


def main():
    p=argparse.ArgumentParser(description="Standalone Plateau Dynamics causal/live engine")
    p.add_argument("path"); p.add_argument("--window",type=int,default=9); p.add_argument("--pi-min",type=float,required=True)
    p.add_argument("--time-column"); p.add_argument("--signal-column"); p.add_argument("--eps-d",type=float); p.add_argument("--eps-v",type=float)
    p.add_argument("--persistence",choices=["duration","samples"],default="duration"); p.add_argument("--calibration-points",type=int,default=256); p.add_argument("--catalogue")
    a=p.parse_args(); L=Lambda(a.window,a.pi_min,a.eps_d,a.eps_v,calibration_points=a.calibration_points,persistence=a.persistence)
    e, events=stream_file(a.path,L=L,time_column=a.time_column,signal_column=a.signal_column)
    print(f"streamed={len(e.f)} lens_frozen={e.lens_frozen} plateaus={len(e.plateau_catalogue())}")
    if a.catalogue: pd.DataFrame(e.plateau_catalogue()).to_csv(a.catalogue,index=False)
    for event in events: print(event)


if __name__ == "__main__": main()
