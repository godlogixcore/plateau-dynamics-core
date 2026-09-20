# Plateau Dynamics Core

This is a standalone causal/live reference engine. It is a separate project and
has no dependency on the UPM Engine repository or its modules.

## Install

```bash
pip install -r requirements.txt
```

## Python usage

```python
from plateau_dynamics_core import Lambda, PDEngine

L = Lambda(m=9, pi_min=5.0, eps_d=0.02, eps_v=0.01)
engine = PDEngine(L)
for t, value in zip(times, values):
    snapshot = engine.feed(t, value)
    if snapshot["event"] in {"plateau_on", "plateau_off"}:
        print(snapshot)
```

If thresholds are omitted, they are calibrated once from
`calibration_points` causal operator evaluations and then frozen. Live Order
starts only after confirmation; `result()` additionally exposes distinct
retrospective support.

## Command line

```bash
python plateau_dynamics_core.py signal.csv \
  --signal-column value --time-column time --window 9 --pi-min 5 \
  --catalogue plateaus.csv
```

Supported inputs are CSV/TXT and one-dimensional NPY. Spatial graph analysis is
available through `constitute_spatial`; HEALPix support can be added as an
optional adapter when `healpy` is installed.

This project reports measured temporal/spatial structure only. It does not
assign clinical or domain labels.
