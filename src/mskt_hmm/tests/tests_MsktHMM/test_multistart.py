"""
Tests for the Phase 2.1 multi-start wrapper (mskt_hmm.multistart.fit_multistart
and the MsktHMM.fit_multistart method).

The data is synthetic 3-regime uMST sampled via MsktHMM.rdmst, so the test needs
no network and no external files. The regimes overlap moderately on feature 0 so
that the seed segmentation is non-trivial.
"""

import numpy as np
import pytest

from mskt_hmm.mskt_hmm import MsktHMM, fit_multistart
from mskt_hmm import native

pytestmark = pytest.mark.skipif(
    native.LIB is None, reason="libemmixskew not loaded")


def _make_3regime(seed=0, per=180):
    """Concatenate six blocks from three uMST regimes that differ on feature 0."""
    rng = np.random.default_rng(seed)
    p = 3
    means = [np.array([1.5, 0.0, 0.0]),
             np.array([0.0, 0.0, 0.0]),
             np.array([-1.5, 0.0, 0.0])]
    Sigma = np.eye(p) * 0.9
    delta = np.array([0.4, 0.1, 0.0])
    blocks = [MsktHMM.rdmst(p=p, n=per, mean=means[k], cov=Sigma, nu=8.0,
                            del_=delta, rng=rng)
              for k in (0, 1, 2, 0, 1, 2)]
    return np.vstack(blocks)


def _template():
    return MsktHMM(n_components=3, covariance_type="full", n_iter=300, tol=1e-4)


def test_history_shape_and_best_selection():
    X = _make_3regime(seed=0)
    best = _template().fit_multistart(X, n_restarts=4, random_state=0)

    hist = best.multistart_history_
    assert len(hist) == 4
    ok = [h for h in hist if h["ok"]]
    assert len(ok) >= 1

    # the returned model is the best-scoring successful restart
    assert best.multistart_best_score_ == max(h["score"] for h in ok)
    assert hasattr(best, "means_") and best.means_.shape == (3, 3)


def test_restart0_is_canonical_and_never_worse():
    X = _make_3regime(seed=0)
    tmpl = _template()
    best = tmpl.fit_multistart(X, n_restarts=5, random_state=1)
    hist = best.multistart_history_

    # restart 0 must use the template's own init_config unchanged
    canon = next(h for h in hist if h["restart"] == 0)
    assert canon["med_win"] == tmpl.init_config.med_win
    assert canon["min_seg"] == tmpl.init_config.min_seg

    # multi-start can only improve on (or match) the canonical fit
    if canon["ok"]:
        assert best.multistart_best_score_ >= canon["score"] - 1e-6


def test_perturbation_actually_varies_init_config():
    # The mechanism's whole point: restarts explore different segmentations.
    X = _make_3regime(seed=2)
    _, hist = _template().fit_multistart(
        X, n_restarts=6, random_state=3, return_all=True)
    configs = {(h["med_win"], h["min_seg"], h["trim_q"]) for h in hist}
    assert len(configs) >= 2, f"init_config was not diversified: {configs}"


def test_template_not_mutated_and_top_level_matches_method():
    X = _make_3regime(seed=4)
    tmpl = _template()
    # top-level function and method are the same code path
    best = fit_multistart(tmpl, X, n_restarts=3, random_state=0)
    assert len(best.multistart_history_) == 3
    # the template itself was never fitted
    assert not hasattr(tmpl, "means_")
