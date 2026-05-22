"""
Temporary multi-start wrapper for MsktHMM.

# TODO : Planned improvement is to tune `med_win`, `min_seg`, `trim_q` within mskt_hmm.py, 
and to delete this file.

-------------------------------------------------------------------------

The EM objective for the uMST-HMM is non-convex, so a single fit is at the
mercy of the initializer. This wrapper fits several fresh models from diversified
initializations and keeps the one with the highest training log-likelihood.

Why the diversity comes from `init_config`, not from jittered init values

-------------------------------------------------------------------------
The initial idea was jittering the initial `transmat_`, `startprob_`
and `means_`. That does not work with the *current* MsktHMM, and the reason was
confirmed empirically: the warm-start initializer (`_init_robust_segments`) is
fully deterministic. It does not read `self.random_state`, and it overwrites
`startprob_`, `transmat_` and the emission parameters from its own seed labels
(mskt_hmm.py around lines 1098-1103). So two fits that differ only in
`random_state`, or in pre-set init values, converge to the *identical* solution.

The lever the segment-based initializer genuinely responds to is its
parameters: `med_win`, `min_seg`, `trim_q`. Perturbing those changes the
seed segmentation and therefore the basin of attraction. On real S&P 500 data
this both diversifies the fits and finds higher-likelihood optima than the
single deterministic default. We therefore diversify across `init_config`.


"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from sklearn.utils import check_random_state

from .init_config import InitConfig


def _perturb_init_config(base: InitConfig, rng, scale: float) -> InitConfig:
    """Return a copy of `base` with the segmentation parameters perturbed."""
    lo, hi = 1.0 - scale, 1.0 + scale
    med = int(round(base.med_win * rng.uniform(lo, hi)))
    med = int(np.clip(med, 11, 121))
    if med % 2 == 0:                      # median filter window must be odd
        med += 1
    min_seg = int(np.clip(round(base.min_seg * rng.uniform(lo, hi)), 30, 400))
    trim_q = float(np.clip(base.trim_q + rng.uniform(-0.03, 0.012), 0.92, 0.995))
    return dataclasses.replace(base, med_win=med, min_seg=min_seg, trim_q=trim_q)


def fit_multistart(template, X, lengths=None, *, n_restarts=10,
                   random_state=None, perturb_scale=0.5, return_all=False):
    """Fit `n_restarts` MsktHMM models from diversified inits, return the best.

    Parameters
    ----------
    template : MsktHMM
        Provides the constructor configuration via ``get_params()``. It is not
        mutated and not itself fitted; every restart is a fresh instance, which
        avoids the carryover that reusing one instance would cause (init values,
        frozen labels, etc. are not reset by ``fit``).
    X, lengths : array-like
        Training data, as passed to ``MsktHMM.fit``.
    n_restarts : int
        Number of fits. Restart 0 uses the template's own ``init_config``
        unchanged, so the result is never worse than the plain single fit; the
        rest perturb ``init_config`` (see module docstring).
    random_state : int or RandomState
        Seeds the perturbations, for reproducibility.
    perturb_scale : float
        Relative spread of the multiplicative jitter on ``med_win`` / ``min_seg``
        (0.5 -> roughly +/-50 percent).
    return_all : bool
        If True, return ``(best_model, history)``; otherwise just ``best_model``.

    Returns
    -------
    MsktHMM
        The best-scoring fitted model, with two attributes attached:
        ``multistart_history_`` (list of per-restart dicts) and
        ``multistart_best_score_`` (float).

    Notes
    -----
    The current initializer supports ``n_components == 3`` only (W2.1); restarts
    that fail for any reason are recorded with ``ok=False`` and skipped.
    """
    from .mskt_hmm import MsktHMM   # local import avoids a circular dependency

    rng = check_random_state(random_state)
    params = template.get_params(deep=False)
    base_ic = params.get("init_config") or InitConfig()

    history, best = [], None
    for r in range(n_restarts):
        ic = base_ic if r == 0 else _perturb_init_config(base_ic, rng, perturb_scale)
        kw = dict(params)
        kw["init_config"] = ic
        kw["random_state"] = int(rng.randint(0, 2**31 - 1))
        model = MsktHMM(**kw)

        rec = {"restart": r, "med_win": ic.med_win, "min_seg": ic.min_seg,
               "trim_q": round(ic.trim_q, 4), "n_frozen_iter": ic.n_frozen_iter}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, lengths)
            score = float(model.score(X, lengths))
            if not np.isfinite(score):
                raise FloatingPointError("non-finite training log-likelihood")
            rec.update(score=score, converged=bool(model.monitor_.converged),
                       iters=int(model.monitor_.iter), ok=True)
            if best is None or score > best[0]:
                best = (score, model)
        except Exception as exc:                       # a perturbed init may fail
            rec.update(score=float("-inf"), converged=False, iters=0,
                       ok=False, error=repr(exc))
        history.append(rec)

    if best is None:
        raise RuntimeError(
            "fit_multistart: every restart failed.\n"
            + "\n".join(str(r) for r in history))

    best_score, best_model = best
    best_model.multistart_history_ = history
    best_model.multistart_best_score_ = best_score
    return (best_model, history) if return_all else best_model
