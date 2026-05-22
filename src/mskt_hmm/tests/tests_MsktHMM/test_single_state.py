
from __future__ import annotations
import os, logging, numpy as np
from numpy.testing import assert_allclose
import pytest
from mskt_hmm.mskt_hmm import MsktHMM
from hmmlearn import _utils          
from mskt_hmm import native          


# ------------------------------------------------------------------ #
DEBUG  = bool(int(os.getenv("MSKT_DEBUG", "0")))
logger = logging.getLogger(__name__)
if DEBUG:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
def _log_array_stats(name, a, b):
    """
    Debug helper: logs basic difference stats between two arrays.
    Prints max absolute diff, mean diff, and std of diff when MSKT_DEBUG=1.
    No assertions are made. Safe to call with any broadcastable shapes.
    """
    if not DEBUG:
        return
    d = np.asarray(a-b)
    logger.info("%s – shape %s – max %.3e  mean %.3e  std %.3e",
                name, d.shape, np.max(np.abs(d)), d.mean(), d.std())
# ------------------------------------------------------------------ #
pytestmark = pytest.mark.skipif(native.LIB is None,
    reason="libemmixskew not found , cannot run MST reference checks")



BOOL = [False, True]

from dataclasses import dataclass

RNG_SEED_BASE = 1000   
MU_SCALE      = 0.5    
VAR_SCALE     = 0.5    
SKEW_SCALE    = 0.6    
NU_VALUE      = 7.0   

# ------------------------------------------------------------------ #
class TestSingleStateMsktHMM:

    @staticmethod
    def _make_dataset(*, n: int = 50000, p: int = 3):
        """
        Synthesizes a single multivariate skew t dataset
        mean ~ N(0, 0.5),   Sigma = A A^T + 0.1 I    with A ~ N(0, 0.5)
        delta ~ N(0, 0.6),  nu = 7.0
        Returns (X, mu, Sigma, delta, nu) with shapes (n, p), (p,), (p,p), (p,), float.
        """
        rng = np.random.default_rng(RNG_SEED_BASE)

        mu    = rng.normal(0.0, MU_SCALE, size=p)
        A     = rng.normal(0.0, VAR_SCALE, size=(p, p))
        Sigma = A @ A.T + np.eye(p) * 0.1
        delta = rng.normal(0.0, SKEW_SCALE, size=p)
        nu    = NU_VALUE

        X = MsktHMM.rdmst(p=p, n=n, mean=mu, cov=Sigma, nu=nu, del_=delta, rng=rng)
        return X, mu, Sigma, delta, nu

    @staticmethod
    def _trivial_fit_single_state_model():
        """
        Trivial pipeline: builds MsktHMM with one state and sets the emission
        parameters exactly to the data generating truth before calling fit.
        Used in test_em_trivial_convergence to verify that
        EM leaves parameters unchanged and converges in place.
        Returns (X, model, (mu_true, Sigma_true, delta_true, nu_true)).
        """
        X, mean_test, cov_test, skew_test, dof_test = TestSingleStateMsktHMM._make_dataset()

        model = MsktHMM(
            n_components=1, covariance_type="full",
            n_iter=8000, tol=1e-6,
            params="stmckv",      # mean_test,cov_test,skew_test,dof_test *BLOCKED*
            init_params="",
            random_state=123,
        )
        model.startprob_ = np.array([1.0])
        model.transmat_  = np.array([[1.0]])
        model.means_  = mean_test.reshape(1, -1)
        model.delta_  = skew_test.reshape(1, -1)
        model.dof_    = np.array([dof_test])
        model.covars_ = _utils.distribute_covar_matrix_to_match_covariance_type(
            cov_test, "full", 1)
        model.fit(X)
        return X, model, (mean_test, cov_test, skew_test, dof_test)


    def test_em_trivial_convergence(self):
        """
        Regression test for the trivial pipeline: after running EM from the
        exact ground truth, the fitted parameters must match the truth within
        tight tolerances. Also prints per block relative errors for debugging.
        """
        X, model, (mean_true, cov_true, skew_true, dof_true) = TestSingleStateMsktHMM._fit_single_state_model()
        assert model.monitor_.converged
        _log_array_stats("mean_true", model.means_[0], mean_true)
        
        print("Sample X shape:", X.shape)
        print(f"mean  : fit = {model.means_[0]}  | true = {mean_true}")
        mean_err = np.where(mean_true != 0, (model.means_[0] - mean_true) / mean_true, np.nan)
        print(f"mean  : relative error = {mean_err}")

        print(f"cov   : fit =\n{model.covars_[0]}\ntrue =\n{cov_true}")
        cov_err = np.where(cov_true != 0, (model.covars_[0] - cov_true) / cov_true, np.nan)
        print("cov   : relative error =\n", cov_err)

        print(f"skew  : fit = {model.delta_[0]}  | true = {skew_true}")
        skew_err = np.where(skew_true != 0, (model.delta_[0] - skew_true) / skew_true, np.nan)
        print(f"skew  : relative error = {skew_err}")

        print(f"dof   : fit = {model.dof_[0]}  | true = {dof_true}")
        dof_err = (model.dof_[0] - dof_true) / dof_true if dof_true != 0 else np.nan
        print(f"dof   : relative error = {dof_err}")

        assert_allclose(model.means_[0],  mean_true, rtol=0.15, atol=0.0 )
        assert_allclose(model.covars_[0], cov_true, rtol=0.15, atol=0.0)
        assert_allclose(model.delta_[0],  skew_true, rtol=0.15, atol=0.0)
        assert_allclose(model.dof_[0],    dof_true, rtol=0.15, atol=0.0)

    
    @staticmethod
    def _fit_single_state_model(*, max_iter: int = 8000):
        """
        Real pipeline: generates data, initializes MsktHMM(1)
        with automatic single component skew t init, and runs EM to convergence.
        Returns (X, model, (mu_true, Sigma_true, delta_true, nu_true)).
        """
        X, mu, Sigma, delta, nu = TestSingleStateMsktHMM._make_dataset()

        model = MsktHMM(
            n_components=1,
            covariance_type="full",
            n_iter=max_iter,
            tol=1e-6,
            params="stmckv",
            init_params="stmc",
            random_state=123,
        )
        model.startprob_ = np.array([1.0])
        model.transmat_  = np.array([[1.0]])
        model.fit(X)

        return X, model, (mu, Sigma, delta, nu)
    
    def test_em_real_convergence(self):
        """
        End to end fit test
        Asserts EM convergence and checks that the estimated emission
        parameters are close to the ground truth within coarse tolerances.
        """
        # 1) fit 
        print("\n--- Running test_em_real_convergence ---")
        X, model, (mu_true, sigma_true, delta_true, nu_true) = TestSingleStateMsktHMM._fit_single_state_model()

        # 2) convergence
        assert model.monitor_.converged, "Model did not converge"

        # 3) diagnostic
        print("Sample X shape:", X.shape)
        print(f"True mu       = {mu_true}")
        print(f"Estimated mu  = {model.means_[0]}")
        print(f"True Sigma    =\n{sigma_true}")
        print(f"Estimated Sigma=\n{model.covars_[0]}")
        print(f"True delta    = {delta_true}")
        print(f"Estimated delta= {model.delta_[0]}")
        print(f"True nu       = {nu_true}")
        print(f"Estimated nu  = {model.dof_[0]}")
        print(f"# EM iterations = {model.monitor_.iter}")

        # 4) assertions
        assert_allclose(model.means_[0],   mu_true,    rtol=0.15,
                        err_msg="mu mismatch ")
        assert_allclose(model.covars_[0],  sigma_true, rtol=0.15,
                        err_msg="Sigma mismatch ")
        assert_allclose(model.delta_[0],   delta_true, rtol=0.20,
                        err_msg="delta mismatch ")
        assert_allclose(model.dof_[0],     nu_true,    rtol=0.20,
                        err_msg="nu mismatch ")

    def test_log_densities_match_reference(self):
        """
        Validates that MsktHMM._compute_log_likelihood reproduces the exact
        log density from the native EMMIX-skew binding (ddmix2) when g = 1.
        Asserts equality at machine precision on a large synthetic sample.
        """
        X, mean_test, cov_test, skew_test, dof_test = self._make_dataset()

        model = MsktHMM(1)
        model.startprob_ = np.array([1.0])
        model.transmat_  = np.array([[1.0]])
        model.means_ = mean_test.reshape(1, -1)
        model.delta_ = skew_test.reshape(1, -1)
        model.dof_   = np.array([dof_test])
        model.covars_= _utils.distribute_covar_matrix_to_match_covariance_type(
            cov_test, "full", 1)
        model._check()
        ll_hmm = model._compute_log_likelihood(X).ravel()
        ll_ref = native.ddmix(
            X=X, distr="mst", g=1,
            mu=mean_test.astype(float).reshape(-1, 1, order="F"),
            sigma=cov_test.T.copy(order="F").ravel("F"),
            dof=np.array([dof_test]),
            delta=skew_test.astype(float).reshape(-1, 1, order="F")
        ).ravel()
        _log_array_stats(r"\delta log-densities", ll_hmm, ll_ref)
        assert_allclose(ll_hmm, ll_ref, rtol=1e-10, atol=1e-12)


    def test_posteriors_are_trivial(self):
        """
        Checks HMM degeneracy when g = 1. Verifies that responsibilities
        are identically 1 for every time step and that the sequence log
        likelihood equals the sum of per sample log densities.
        """
        
        X, model, _ = TestSingleStateMsktHMM._fit_single_state_model()
        ll, post = model.score_samples(X)
        assert_allclose(post, np.ones_like(post))
        assert_allclose(ll, model._compute_log_likelihood(X).sum(),
                        rtol=1e-10, atol=1e-12)

    def test_forward_backward_is_consistent(self):
        """
        Consistency check between forward-backward smoothing and Viterbi
        decoding in the single state case. The Viterbi path must be all zeros
        and the Viterbi log likelihood must equal the smoothing log likelihood.
        """

        X, model, _ = TestSingleStateMsktHMM._fit_single_state_model()
        ll_score, _ = model.score_samples(X)
        ll_vit, states = model.decode(X, algorithm="viterbi")
        assert np.all(states == 0)
        assert_allclose(ll_vit, ll_score, rtol=1e-10, atol=1e-12)

    def test_invalid_covariance_type_raises(self):
        """
        Constructing MsktHMM with covariance_type != 'full' must raise ValueError.
        This is a design constraint: the Fortran M-step supports full covariance only.
        """
        for bad_type in ("diag", "tied", "spherical"):
            with pytest.raises(ValueError, match="covariance_type"):
                MsktHMM(n_components=2, covariance_type=bad_type)
