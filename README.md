# Plateau Dynamics Manuscript Engine

This repository is the standalone **Plateau Dynamics Manuscript Engine**.

It is independent from the UPM Engine repository and contains two separate
implementations:

- `plateau_dynamics_core.py` — compact causal/live core.
- `plateau_dynamics_manuscript_engine.py` — manuscript-grade reference engine
  implementing the full formalism described in the Plateau Dynamics manuscript.

## Manuscript

**Plateau Dynamics: A Unified Operational Definition for Real-Time Detection
and Measurement of Persistent Low-Dynamical Objects in Observation Space**

Author: Ruben Kabongo, 2026  
DOI: `10.5281/zenodo.19781076`

## Manuscript engine capabilities

The manuscript engine provides:

- retrospective temporal constitution;
- fully causal live and streaming operation;
- frozen threshold lenses and calibration prefixes;
- persistence-confirmed Plateau identity;
- post-identity live Order measurement;
- generic spatial graph constitution;
- optional HEALPix adjacency through `healpy`;
- cross-scale lineage helpers;
- CSV and JSON export helpers; and
- a command-line interface for temporal, live, and HEALPix workflows.

The manuscript reference implementation is intentionally separate from the
compact core. Neither implementation depends on the UPM Engine repository.

## Installation

```bash
pip install -r requirements.txt
```

The required dependencies are NumPy, pandas, and SciPy. HEALPix support is
optional:

```bash
pip install healpy
```

## Usage

```python
from plateau_dynamics_manuscript_engine import Lambda, constitute_temporal

contract = Lambda(m=9, pi_min=5.0, eps_d=0.02, eps_v=0.01)
result = constitute_temporal(values, times, contract)
```

For causal replay:

```python
from plateau_dynamics_manuscript_engine import Lambda, replay_live

live = replay_live(values, times, Lambda(m=9, pi_min=5.0))
```

Command-line examples:

```bash
python plateau_dynamics_manuscript_engine.py temporal signal.csv \
  --signal-column value --time-column time --window 9 --pi-min 5

python plateau_dynamics_manuscript_engine.py live signal.csv \
  --signal-column value --time-column time --window 9 --pi-min 5
```

The engine reports measured temporal and spatial structure only. It does not
assign clinical or domain labels.
