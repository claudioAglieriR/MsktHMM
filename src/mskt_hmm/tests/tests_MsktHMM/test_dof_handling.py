"""
Tests for Phase 2.2 DoF handling:
  - `dof_upper` is a real constructor argument (exposed via get_params, threaded
    into the M-step as the nu cap);
  - the per-state Mardia freeze pins nu at `dof_upper` for Gaussian-tailed
    regimes and leaves it adaptive for heavy-tailed ones.

Data is synthetic 3-regime, generated two ways (Gaussian vs heavy-tailed
skew-t), so the test needs no network or external files.
"""

import numpy as np
import pytest

from mskt_hmm.mskt_hmm import MsktHMM
from mskt_hmm import native

pytestmark = pytest.mark.skipif(
    native.LIB is None, reason="libemmixskew not loaded")

_MEANS = [np.array([4.0, 0.0, 0.0]),
          np.array([0.0, 0.0, 0.0]),
          np.array([-4.0, 0.0, 0.0])]


def _three_regime(kind, seed=0, per=400):
    """Six concatenated blocks from three regimes separated on feature 0."""
    rng = np.random.default_rng(seed)
    Sigma = np.eye(3) * 1.0
    blocks = []
    for k in (0, 1, 2, 0, 1, 2):
        if kind == "gauss":
            Xk = rng.multivariate_normal(_MEANS[k], Sigma, size=per)
        else:  # heavy-tailed, skewed skew-t (nu = 4)
            Xk = MsktHMM.rdmst(p=3, n=per, mean=_MEANS[k], cov=Sigma, nu=4.0,
                               del_=np.array([1.5, 0.5, 0.0]), rng=rng)
        blocks.append(Xk)
    return np.vstack(blocks)


def test_dof_upper_exposed_and_in_params():
    m = MsktHMM(n_components=3, dof_upper=50.0)
    assert m.dof_upper == 50.0
    assert m.get_params()["dof_upper"] == 50.0   # sklearn-visible -> survives clone/multistart


def test_freeze_pins_nu_on_gaussian_regimes():
    X = _three_regime("gauss", seed=1)
    m = MsktHMM(n_components=3, n_iter=200, tol=1e-4, dof_upper=80.0)
    m.fit(X)
    mask = m._nu_frozen_mask_
    assert mask.any(), "expected at least one Gaussian regime to freeze nu"
    # frozen states are pinned exactly at the cap, not merely near it
    assert np.allclose(m.dof_[mask], 80.0)


def test_no_freeze_on_heavy_tailed_regimes():
    X = _three_regime("heavy", seed=2)
    m = MsktHMM(n_components=3, n_iter=300, tol=1e-4)
    m.fit(X)
    mask = m._nu_frozen_mask_
    assert not mask.any(), f"heavy-tailed regimes should stay adaptive, got {mask}"
    # and the recovered nu sits well below the cap (true nu was 4)
    assert np.any(m.dof_ < 50.0)


def test_dof_upper_caps_nu():
    # A low cap must bound nu even on Gaussian data (where the MLE diverges).
    X = _three_regime("gauss", seed=3)
    m = MsktHMM(n_components=3, n_iter=200, tol=1e-4, dof_upper=30.0)
    m.fit(X)
    assert np.all(m.dof_ <= 30.0 + 1e-6)
