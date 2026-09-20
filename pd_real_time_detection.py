#!/usr/bin/env python3
"""PD_ Real-Time Detection — public runtime integration layer.

The default scientific backend is ``plateau_dynamics_manuscript_engine``.
Override it with the ``PD_ENGINE_MODULE`` environment variable when deploying
a compatible private backend.
"""
from __future__ import annotations

import argparse
import heapq
import importlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RuntimeConfig:
    engine_kwargs: Dict[str, Any] = field(default_factory=lambda: {"m": 9, "persistence": "samples"})
    engine_module: Optional[str] = None
    max_lateness_s: float = 0.0
    late_policy: str = "drop"
    duplicate_policy: str = "drop"
    keep_trace: bool = True
    keep_events: bool = True

    def validate(self) -> "RuntimeConfig":
        if self.max_lateness_s < 0: raise ValueError("max_lateness_s must be >= 0")
        if self.late_policy not in {"drop", "raise"}: raise ValueError("invalid late_policy")
        if self.duplicate_policy not in {"drop", "raise"}: raise ValueError("invalid duplicate_policy")
        return self


@dataclass
class Sample:
    t: float
    value: float
    source: str = "external"
    seq: int = 0
    symbol: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def parse_timestamp(value: Any) -> float:
    if value is None: return time.time()
    if isinstance(value, (int, float, np.integer, np.floating)):
        x = float(value)
        if not np.isfinite(x): raise ValueError("timestamp must be finite")
        if abs(x) > 1e17: return x / 1e9
        if abs(x) > 1e14: return x / 1e6
        if abs(x) > 1e11: return x / 1e3
        return x
    text = str(value).strip().replace("Z", "+00:00")
    try: return parse_timestamp(float(text))
    except ValueError: pass
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def _load_backend(name: Optional[str] = None):
    """Load the manuscript engine by default, with configurable fallbacks."""
    requested = name or os.environ.get("PD_ENGINE_MODULE") or "plateau_dynamics_manuscript_engine"
    tried: List[str] = []
    for candidate in (requested, "plateau_dynamics_manuscript_engine", "plateau_dynamics_core"):
        if candidate in tried: continue
        tried.append(candidate)
        try: module = importlib.import_module(candidate)
        except Exception: continue
        if hasattr(module, "Lambda") and hasattr(module, "PDEngine"): return module
    raise RuntimeError(
        "No compatible PD backend found. Expected "
        "plateau_dynamics_manuscript_engine or plateau_dynamics_core, "
        "or set PD_ENGINE_MODULE."
    )


class OrderedSampleBuffer:
    def __init__(self, *, max_lateness_s=0.0, late_policy="drop", duplicate_policy="drop"):
        self.lateness = float(max_lateness_s); self.late_policy = late_policy; self.duplicate_policy = duplicate_policy
        self.heap=[]; self.max_seen=-np.inf; self.last=-np.inf; self.counter=0
        self.stats={"received":0,"emitted":0,"late_dropped":0,"duplicate_dropped":0}

    def push(self, sample: Sample) -> List[Sample]:
        self.stats["received"] += 1; t=sample.t
        if t < self.last or t == self.last:
            late=t < self.last; policy=self.late_policy if late else self.duplicate_policy
            if policy == "raise": raise ValueError(f"invalid event-time sample {t}")
            self.stats["late_dropped" if late else "duplicate_dropped"] += 1; return []
        self.counter += 1; self.max_seen=max(self.max_seen,t); heapq.heappush(self.heap,(t,self.counter,sample))
        return self._pop(self.max_seen-self.lateness)

    def _pop(self, watermark):
        out=[]
        while self.heap and self.heap[0][0] <= watermark:
            t,_,sample=heapq.heappop(self.heap)
            if t <= self.last: continue
            self.last=t; self.stats["emitted"] += 1; out.append(sample)
        return out

    def flush(self): return self._pop(np.inf)


class PDRealTimeDetection:
    """Application-facing runtime around a Plateau Dynamics backend."""
    PUBLIC_EVENTS={"candidate_on","candidate_reject","confirm","exit","lens_frozen"}

    def __init__(self, config: Optional[RuntimeConfig]=None):
        self.config=(config or RuntimeConfig()).validate(); self.backend=_load_backend(self.config.engine_module)
        self.L=self.backend.Lambda(**self.config.engine_kwargs); self.engine=self.backend.PDEngine(self.L)
        self.buffer=OrderedSampleBuffer(max_lateness_s=self.config.max_lateness_s, late_policy=self.config.late_policy, duplicate_policy=self.config.duplicate_policy)
        self.callbacks=[]; self._updates=[]; self._events=[]; self._last=None

    def on_update(self, callback: Callable[[Dict[str,Any]],None]): self.callbacks.append(callback); return self

    def push(self, value, *, t=None, source="external", symbol=None, meta=None):
        return self.push_sample(Sample(parse_timestamp(t), float(value), source=source, symbol=symbol, meta=dict(meta or {})))

    def push_sample(self, sample: Sample):
        if not np.isfinite(sample.value): raise ValueError("sample value must be finite")
        return [self._feed(item) for item in self.buffer.push(sample)]

    def flush(self): return [self._feed(item) for item in self.buffer.flush()]

    def run(self, source: Iterable[Sample], limit: Optional[int]=None) -> Iterator[Dict[str,Any]]:
        count=0
        for sample in source:
            for update in self.push_sample(sample):
                yield update; count += 1
                if limit is not None and count >= limit: self.flush(); return
        yield from self.flush()

    def _feed(self, sample):
        snapshot=self.engine.feed(sample.t,sample.value)
        output={"t":sample.t,"value":sample.value,"ready":bool(snapshot.get("ready",False)),"state":str(snapshot.get("state","warmup")),"event":str(snapshot.get("event","warmup")),"regime_active":bool(snapshot.get("confirmed_membership",False)),"order":float(snapshot.get("O_live",0) or 0),"plateau_id":snapshot.get("current_plateau_id"),"source":sample.source,"symbol":sample.symbol}
        if sample.meta: output["meta"]=sample.meta
        if self.config.keep_trace: self._updates.append(dict(output))
        if self.config.keep_events and output["event"] in self.PUBLIC_EVENTS: self._events.append(dict(output))
        self._last=output
        for callback in self.callbacks: callback(dict(output))
        return output

    def state(self): return dict(self._last) if self._last else {"ready":False,"state":"warmup","event":"warmup","regime_active":False,"order":0.0,"plateau_id":None}
    def updates(self): return pd.DataFrame(self._updates)
    def events(self): return pd.DataFrame(self._events)
    def ordering_stats(self): return dict(self.buffer.stats)

    def replay_file(self, path, *, time_column=None, value_column=None):
        df=pd.read_csv(path)
        if value_column is None:
            numeric=df.select_dtypes(include=[np.number]).columns.tolist(); value_column=next(c for c in numeric if c != time_column)
        times=df[time_column] if time_column else range(len(df))
        for i,(timestamp,value) in enumerate(zip(times,df[value_column])): self.push(float(value),t=timestamp,source="file")
        self.flush(); return self.updates()

    def export(self,prefix):
        p=Path(prefix); p.parent.mkdir(parents=True,exist_ok=True)
        paths={"trace":str(p)+"_runtime_trace.csv","events":str(p)+"_runtime_events.csv","state":str(p)+"_runtime_state.json"}
        self.updates().to_csv(paths["trace"],index=False); self.events().to_csv(paths["events"],index=False); Path(paths["state"]).write_text(json.dumps(self.state(),indent=2,default=str))
        return paths


def create_runtime(*, engine_kwargs=None, **kwargs):
    return PDRealTimeDetection(RuntimeConfig(engine_kwargs=dict(engine_kwargs or {"m":9,"persistence":"samples"}), **kwargs))


def main(argv=None):
    parser=argparse.ArgumentParser(description="PD_ Real-Time Detection")
    parser.add_argument("path"); parser.add_argument("--time-column"); parser.add_argument("--value-column"); parser.add_argument("--window",type=int,default=9); parser.add_argument("--persistence",choices=["samples","time"],default="samples"); parser.add_argument("--export-prefix")
    args=parser.parse_args(argv); runtime=create_runtime(engine_kwargs={"m":args.window,"persistence":args.persistence}); runtime.replay_file(args.path,time_column=args.time_column,value_column=args.value_column)
    print(json.dumps(runtime.state(),default=str))
    if args.export_prefix: print(json.dumps(runtime.export(args.export_prefix),indent=2))


if __name__ == "__main__": main()
