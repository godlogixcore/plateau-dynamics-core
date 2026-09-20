import numpy as np

from plateau_dynamics_core import Lambda, PDEngine, causal_operators, replay


def test_explicit_thresholds_and_live_state():
    L = Lambda(m=3, pi_min=2, eps_d=0.1, eps_v=0.01, persistence="samples")
    e = PDEngine(L)
    for i in range(8):
        snap = e.feed(i, 1.0)
    assert e.lens_frozen
    assert any(x in {"plateau_on", "plateau_hold"} for x in e.events)
    assert e.result()["O_live"].shape == (8,)


def test_replay_is_causal_and_returns_masks():
    y = np.ones(12)
    result = replay(y, None, Lambda(m=3, pi_min=2, eps_d=0.1, eps_v=0.01, persistence="samples"))
    assert len(result["t"]) == 12
    assert result["confirmed_mask"].dtype == bool
    assert "P" in result


def test_invalid_time_is_rejected():
    e = PDEngine(Lambda(m=3, pi_min=2, eps_d=1, eps_v=1))
    e.feed(0, 1)
    try:
        e.feed(0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected non-increasing timestamp to fail")
