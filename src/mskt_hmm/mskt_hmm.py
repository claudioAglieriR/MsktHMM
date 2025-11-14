import numpy as np
from . import _emissions, _utils
from .base import BaseHMM
from .utils import fill_covars
from . import _hmmc
import importlib, src.mskt_hmm.native as native
from sklearn.utils import check_random_state
from hmmlearn.base import ConvergenceMonitor
from hmmlearn.base import ConvergenceMonitor
import numpy as np
import logging
import numpy as np
from . import native, _utils  
from sklearn.utils import check_random_state
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import gamma
import sys


_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from sklearn.utils.validation import (
     check_random_state)

importlib.reload(native)

__all__ = [
    "MsktHMM",
]

_log = logging.getLogger(__name__)
COVARIANCE_TYPES = frozenset(("full"))

class BaseMsktHMM(_emissions._AbstractHMM):

    def __init__(self, n_components=1,
                startprob_prior=1.0, transmat_prior=1.0,
                algorithm="viterbi", random_state=None,
                n_iter=8000, tol=1e-2, verbose=False,
                params="stmckv", init_params="stmc",
                implementation="log"):
        """
        Initialize a Hidden Markov Model with multivariate skew-t (uMST) emissions.

        This base class wires up the discrete HMM mechanics (startprob, transmat,
        decoding, forward-backward) while leaving emission specifics to the
        uMST methods implemented here. Training uses EM with either log-space
        or scaling forward-backward.

        Parameters
        ----------
        n_components : int
            Number of hidden states.
        startprob_prior : float or array-like
            Dirichlet prior for the initial state distribution.
        transmat_prior : float or array-like
            Dirichlet prior for each row of the transition matrix.
        algorithm : {"viterbi", "map"}
            Decoder to produce a state path during `decode`.
        random_state : int or RandomState
            Random seed or RNG instance for reproducibility.
        n_iter : int
            Maximum number of EM iterations.
        tol : float
            Convergence tolerance on the EM lower bound.
        verbose : bool
            Whether to print convergence reports via the monitor.
        params : str
            Which parameter blocks to update during EM.
            Supported letters: "s" (startprob), "t" (transmat), "m" (means),
            "c" (covars), "k" (skewness), "v" (degrees of freedom).
            With current implementation, skewness and degrees of freedom
            always updated
        init_params : str
            Which parameter blocks to initialize before EM.
        implementation : {"log", "scaling"}
            Numerical implementation of forward-backward.
        """

        super().__init__(n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior, algorithm=algorithm,
                         random_state=random_state, n_iter=n_iter,
                         tol=tol, params=params, verbose=verbose,
                         init_params=init_params,
                         implementation=implementation)


    def _get_n_fit_scalars_per_param(self):
        """
        Return a dict with the number of free scalar parameters per block.

        The counts are used by model selection criteria and sanity checks.
        Blocks follow the same letters used in `params` and `init_params`:
        s: startprob (n_components - 1)
        t: transmat  (n_components * (n_components - 1))
        m: means     (n_components * n_features)
        c: covars    (n_components * n_features * (n_features + 1) // 2)
        k: skewness  (n_components * n_features)
        v: dof       (n_components)

        Returns
        -------
        dict
            Map from block letter to integer count.
        Raises
        ------
        ValueError
            If the covariance_type is unsupported.
        """

        nc = self.n_components
        nf = self.n_features

        cov_counts = {
            "full": nc * nf * (nf + 1) // 2,  # \Sigma_k 
        }
        if self.covariance_type not in cov_counts:
            raise ValueError(f"Unsupported covariance_type '{self.covariance_type}'")

        return {
            "s": nc - 1,  # start-probabilities
            "t": nc * (nc - 1),  # transition matrix
            "m": nc * nf,  # means
            "c": cov_counts[self.covariance_type],
            "k": nc * nf,  # skew-vectors \delta_k
            "v": nc,  # degrees of freedom \nu_k
        }


    def _initialize_sufficient_statistics(self):
        """
        Allocate and zero the EM sufficient-statistics container for uMST emissions.

        In addition to the standard HMM fields ('nobs', 'start', 'trans'),
        this creates emission-level accumulators used by the M-step:
        post[k]      = sum_i gamma_{i,k}
        ev_sum[k]    = sum_i gamma_{i,k} * E[V | y_i, k]
        ez2_sum[k]   = sum_i gamma_{i,k} * E[Z^2 | y_i, k]
        obs_ev[k,:]  = sum_i gamma_{i,k} * E[V | y_i, k] * y_i
        obs_ez1[k,:] = sum_i gamma_{i,k} * E[|Z| | y_i, k] * y_i
        yy_ev[k,:,:] or obs2_ev[k,:] depending on covariance_type

        Notes
        -----
        V and Z are the standard latent variables in the Sahu-Dey-Branco uMST
        representation. See researche article 'EMMIXcskew: An R Package for the 
        Fitting of a Mixture of Canonical Fundamental Skew t-Distributions'
        by Lee and McLachan for further explanations.
        Their conditional moments are produced during the E-step
        and cached for accumulation here.
        """

        stats = super()._initialize_sufficient_statistics()

        nc, nf = self.n_components, self.n_features

        # -----------------------------------------------------------------
        # scalars
        # -----------------------------------------------------------------
        stats["post"] = np.zeros(nc)  # \sum gamma
        stats["ev_sum"] = np.zeros(nc)  # \sum gamma*E[V]
        stats["ez2_sum"] = np.zeros(nc)  # \sum gamma*E[Z ^2]

        # -----------------------------------------------------------------
        # vectors
        # -----------------------------------------------------------------
        stats["obs_ev"] = np.zeros((nc, nf))  # \sum gamma*E[V]*y
        stats["obs_ez1"] = np.zeros((nc, nf))  # \sum gamma*E[|Z|]*y

        # -----------------------------------------------------------------
        # matrix
        # -----------------------------------------------------------------
        if self.covariance_type in ("full"):
            stats["yy_ev"] = np.zeros((nc, nf, nf))  # \sum gamma*E[V]*y y 
        else:  
            stats["obs2_ev"] = np.zeros((nc, nf))  # \sum gamma*E[V]*y ^2

        return stats


    def _compute_log_likelihood(self, X):
        """
        Compute per-sample, per-state log emission likelihoods under uMST.

        This method flattens the current parameters into Fortran order and calls
        the native 'denmst' routine to evaluate:
        logdens[i, k] = log p(y_i | state=k, uMST params)
        It also receives conditional moments of the latent variables and caches
        them in 'self._mst_cache' for later use in '_accumulate_sufficient_statistics'.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observations.

        Returns
        -------
        logdens : ndarray, shape (n_samples, n_components)
            Log-likelihood matrix for the current emission parameters.

        Also does
        ------------
        Sets 'self._mst_cache = (ev, ez1v, ez2v, elnv)', each with shape (n_samples, n_components).
        """
        # --- 0. cast -----------------------------------------------------------
        X = np.asarray(X, dtype=np.float64, order="C")
        _, p = X.shape
        g = self.n_components

        # --- 1. parameters flatten ---------------------------------------------
        mu_flat = self.means_.T.ravel("F")

        full_cov = _utils.covars_to_full(self._covars_,
                                         self.covariance_type,
                                         n_components=g,
                                         n_features=p)
        sigma_flat = full_cov.transpose(2, 1, 0).ravel("F")

        delta_flat = self.delta_.T.ravel("F")
        dof = self.dof_.astype(np.float64, copy=False)

        # --- 2. native call -------------------------------------
        logdens, ev, ez1v, ez2v, elnv = native.denmst(
            X=X, distr="mst",
            mu=mu_flat, sigma=sigma_flat, dof=dof, delta=delta_flat, g=g
        )

        self._mst_cache = (ev, ez1v, ez2v, elnv)  # shape (n, g) for each one
        return logdens  # shape (n, g)

    def _do_estep(self, X, lengths):
        """
        Run one E-step: forward-backward + accumulation of sufficient statistics.

        For each subsequence in (X, lengths), this:
        1) computes log-likelihoods and runs forward-backward (log or scaling),
        2) optionally overrides posteriors with frozen one-hot labels when
            a warm-start "label freeze" is active,
        3) calls '_accumulate_sufficient_statistics' to gather emission and
            transition counts,
        4) sums the total log-likelihood across sequences.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Concatenated observations.
        lengths : array-like of int
            Lengths of the individual sequences.

        Returns
        -------
        stats : dict
            Sufficient statistics for the M-step.
        curr_logprob : float
            Total log-likelihood of the data under the current parameters.

        Side Effects
        ------------
        Decrements and clears the internal '_frozen_labels_' counter when in use.
        """
        impl = {"scaling": self._fit_scaling, "log": self._fit_log}[self.implementation]

        stats = self._initialize_sufficient_statistics()
        self._estep_begin()
        curr_logprob = 0.0

        frozen = getattr(self, "_frozen_labels_", None)
        if frozen is not None:
            lab_vec = frozen["labels"]
            cursor = 0  

        for sub_X in _utils.split_X_lengths(X, lengths):
            lattice, logprob, posteriors, fwdlattice, bwdlattice = impl(sub_X)

            # ----------------------------------------------------
            #  substitute posteriors if frozen
            # ----------------------------------------------------
            if frozen is not None:
                lab_slice = lab_vec[cursor: cursor + len(sub_X)]
                cursor += len(sub_X)
                # one-hot: shape (T, n_components)
                posteriors = np.eye(self.n_components, dtype=float)[lab_slice]

            self._accumulate_sufficient_statistics(
                stats, sub_X, lattice, posteriors, fwdlattice, bwdlattice
            )
            curr_logprob += logprob

        # ---- fine E-step: update counting for frozing procedure ----
        if frozen is not None:
            frozen["remain"] -= 1
            if frozen["remain"] <= 0:
                del self._frozen_labels_

        return stats, curr_logprob

    def _accumulate_sufficient_statistics(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        """
        Accumulate discrete and emission-level sufficient statistics for uMST.

        First delegates to the base HMM to update:
        - stats["nobs"], stats["start"], stats["trans"].

        Then it reads the cached E-step moments from 'self._mst_cache' and:
        - appends block-wise arrays for X, posteriors, and moments
            (keys: "X_blocks", "tau_blocks", "ev_blocks", "ez1_blocks",
                "ez2_blocks", "lnv_blocks"),
        - updates scalar sums for each state: "post", "sumvt", "sumzt", "sumlnv",
        - if means are being updated, accumulates first-order cross terms
            into "y_ev" for the M-step closed forms.

        Parameters
        ----------
        stats : dict
            Sufficient-statistics container (modified in place).
        X : ndarray, shape (T, n_features)
            Current subsequence.
        lattice : ndarray
            Emission log-likelihoods; kept for interface parity.
        posteriors : ndarray, shape (T, n_components)
            State posteriors gamma_t(k).
        fwdlattice, bwdlattice : ndarray
            Forward and backward lattices; used by the base class for transitions.

        Raises
        ------
        RuntimeError
            If '_mst_cache' is missing (E-step moments not computed).
        """

        # 1. update discrete part (\pi, A)
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, posteriors, fwdlattice, bwdlattice)

        g, p = self.n_components, self.n_features

        # ---------------------------------------------------------------
        # 2. retrieve moments
        # ---------------------------------------------------------------
        try:
            ev, ez1v, ez2v, elnv = self._mst_cache
        except (AttributeError, TypeError, ValueError):
            raise RuntimeError("_mst_cache absent ")

        # empty it after usage
        self._mst_cache = None

        # ---------------------------------------------------------------
        # 3. blocks for Fortran M-step
        # ---------------------------------------------------------------
        stats.setdefault("X_blocks", []).append(X)
        stats.setdefault("tau_blocks", []).append(posteriors)
        stats.setdefault("ev_blocks", []).append(ev)
        stats.setdefault("ez1_blocks", []).append(ez1v)
        stats.setdefault("ez2_blocks", []).append(ez2v)
        stats.setdefault("lnv_blocks", []).append(elnv)

        # ---------------------------------------------------------------
        # 4. scalar sums
        # ---------------------------------------------------------------
        stats.setdefault("post", np.zeros(g))
        stats.setdefault("sumvt", np.zeros(g))
        stats.setdefault("sumzt", np.zeros(g))
        stats.setdefault("sumlnv", np.zeros(g))

        stats["post"] += posteriors.sum(axis=0)
        stats["sumvt"] += (posteriors * ev).sum(axis=0)
        stats["sumzt"] += (posteriors * ez2v).sum(axis=0)
        stats["sumlnv"] += (posteriors * elnv).sum(axis=0)

        # ---------------------------------------------------------------
        # 5. first order matrices (mu, delta)
        # ---------------------------------------------------------------
        if self._needs_sufficient_statistics_for_mean():
            stats.setdefault("y_ev", np.zeros((g, p)))
            stats["y_ev"] += (posteriors * ev).T @ X




class SkewtMonitor(ConvergenceMonitor):
    """
    Implement same convergence of emmst_/emskewfit2 for comparison.
    Monitor method used in the implementation of the researche article 
    'EMMIXcskew: An R Package for the Fitting of a Mixture of Canonical 
    Fundamental Skew t-Distributions'by Lee and McLachan.
    """

    @property
    def converged(self):
        if self.iter == self.n_iter:
            return True
        if self.iter < 10:
            return False
        rel10 = abs(self.history[-1] - self.history[-10]) < abs(self.history[-10]) * self.tol
        rel1 = abs(self.history[-1] - self.history[-2]) < abs(self.history[-2]) * self.tol
        return rel10 and rel1
    
    def report(self, log_prob):
        """
        Report convergence progress and record the current lower bound.

        This monitor prints a one-line progress report (iteration, log-probability,
        and delta from the previous iteration) when `verbose=True`, and warns if the
        objective decreases by more than floating-point wiggle room.

        Warm-up silence
        ---------------
        The first three EM iterations are intentionally kept *silent* because
        emission parameters are frozen during the warm-start phase. During these
        iterations we suppress both:
        - the verbose progress line, and
        - the "not converging" warning.
        The history is still recorded so that deltas are correct once reporting
        resumes.

        Parameters
        ----------
        log_prob : float
            The log probability (lower bound) computed at the current iteration.
        """
        # Suppress user-facing convergence messages in the first 3 iterations
        # (self.iter counts from 0 inside the EM loop).
        silent_warmup = (self.iter < 3)

        if self.verbose and not silent_warmup:
            delta = log_prob - self.history[-1] if self.history else np.nan
            message = self._template.format(
                iter=self.iter + 1, log_prob=log_prob, delta=delta)
            print(message, file=sys.stderr)

        # Allow for some wiggle room based on precision; only warn after warm-up.
        precision = np.finfo(float).eps ** 0.5
        if self.history and not silent_warmup:
            if (log_prob - self.history[-1]) < -precision:
                delta = log_prob - self.history[-1]
                _log.warning(
                    "Model is not converging. Current: %s is not greater than %s. "
                    "Delta is %s , iter is %s",
                    log_prob, self.history[-1], delta, self.iter
                )

        # Always record the history and advance the iteration counter.
        self.history.append(log_prob)
        self.iter += 1



class MsktHMM(BaseMsktHMM, BaseHMM):


    def __init__(self, n_components=1, covariance_type='full',
                min_covar=1e-3, startprob_prior=1.0, transmat_prior=1.0,
                means_prior=0, means_weight=0, covars_prior=1e-2,
                covars_weight=1, algorithm="viterbi", random_state=None,
                n_iter=8000, tol=1e-6, verbose=False,
                params="stmckv", init_params="stmc",
                implementation="log"):
        """
        Initialize a multivariate skew t HMM with full covariance emissions.

        Parameters
        ----------
        n_components : int
            Number of hidden states.
        covariance_type : str
            Must be "full". Kept as a parameter for API parity.
        min_covar : float
            Diagonal floor added to covariances for numerical stability.
        startprob_prior, transmat_prior : float or array-like
            Dirichlet priors for initial state distribution and transition rows.
        means_prior, means_weight : float or array-like
            Normal prior mean and precision for emission means.
        covars_prior, covars_weight : float or array-like
            Inverse Wishart prior parameters for covariances when relevant.
        algorithm : {"viterbi","map"}
            Decoder used by decode.
        random_state : int or RandomState
            RNG seed or instance.
        n_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance.
        verbose : bool
            If True, print per iteration progress messages.
        params, init_params : str
            Which parameter groups to update or initialize.
        implementation : {"log","scaling"}
            Forward-backward numerical implementation.

        Notes
        -----
        Sets a custom convergence monitor tailored to skew t emissions.
        """

        super().__init__(n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior, algorithm=algorithm,
                         random_state=random_state, n_iter=n_iter,
                         tol=tol, params=params, verbose=verbose,
                         init_params=init_params,
                         implementation=implementation)
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.monitor_ = SkewtMonitor(self.tol, self.n_iter, self.verbose)
    
    
    @property
    def covars_(self):
        """Return covars as a full matrix."""
        return fill_covars(self._covars_, self.covariance_type,
                           self.n_components, self.n_features)

    @covars_.setter
    def covars_(self, covars):
        covars = np.array(covars, copy=True)
        _utils._validate_covars(covars, self.covariance_type,
                                self.n_components)
        self._covars_ = covars
        

    @staticmethod
    def _rdmvn(n, p, mean=None, cov=None, rng=None):
        """
        Draw samples from a p-dimensional multivariate normal.

        This is a lightweight NumPy equivalent of R's rdmvn. It returns n samples
        from N_p(mean, cov) using numpy.random.Generator.multivariate_normal.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        p : int
            Dimension of the multivariate normal.
        mean : array-like of shape (p,), optional
            Mean vector. Defaults to zeros if None.
        cov : array-like of shape (p, p), optional
            Covariance matrix. Defaults to identity if None.
        rng : numpy.random.Generator, optional
            Random generator. If None, a new default generator is created.

        Returns
        -------
        ndarray of shape (n, p)
            The sampled matrix where each row is one draw.
        """

        if rng is None:
            rng = np.random.default_rng()
        if mean is None:
            mean = np.zeros(p)
        if cov is None:
            cov = np.eye(p)

        return rng.multivariate_normal(mean, cov, size=n)



    @staticmethod
    def rdmst(p, n=5000, mean=None, cov=None, nu=10, del_=None, rng=None):
        """
        Generate samples from the unrestricted multivariate skew-t (uMST, Sahu–Dey–Branco).

        Hierarchical construction:
            W ~ Gamma(nu/2, rate = nu/2)
            eps | W ~ N_p(0, cov / W)
            Z   | W ~ N(0, 1 / W)            # scalar per sample
            Y = mean + eps + |Z| * del_

        Parameters
        ----------
        p : int
            Dimension of the random vector.
        n : int, default=5000
            Number of samples to generate.
        mean : array-like of shape (p,), optional
            Location vector. Defaults to zeros if None.
        cov : array-like of shape (p, p), optional
            Scale (covariance) matrix. Defaults to identity if None.
        nu : float, default=10
            Degrees of freedom (> 0). Smaller values give heavier tails.
        del_ : array-like of shape (p,), optional
            Skewness vector. Defaults to zeros if None.
        rng : numpy.random.Generator, optional
            Random generator. If None, a new default generator is created.

        Returns
        -------
        ndarray of shape (n, p)
            Samples from the multivariate skew-t distribution.

        Notes
        -----
        The Gamma parameterization uses shape = nu/2 and rate = nu/2, so E[W] = 1.
        Conditional independence: eps and Z are independent given W.
        """
        if rng is None:
            rng = np.random.default_rng()

        mean = np.zeros(p) if mean is None else np.asarray(mean, float)
        cov  = np.eye(p)   if cov  is None else np.asarray(cov,  float)
        del_ = np.zeros(p) if del_ is None else np.asarray(del_, float)

        # 1) W ~ Gamma(shape=nu/2, rate=nu/2)  -> numpy uses 'scale' = 1/rate
        W = rng.gamma(shape=nu/2.0, scale=1.0/(nu/2.0), size=n)
        sqrt_W = np.sqrt(W)

        # 2) eps | W  ~ N_p(0, cov / W)
        eps = rng.multivariate_normal(np.zeros(p), cov, size=n) / sqrt_W[:, None]

        # 3) Z | W  ~ N(0, 1/W)  (scalar per sample), take absolute value
        z = np.abs(rng.standard_normal(size=n) / sqrt_W)

        # 4) Y = mean + eps + |Z| * del
        Y = mean[None, :] + eps + z[:, None] * del_[None, :]
        return Y


    @staticmethod
    def _rle(vec: np.ndarray):
        """
        Run-length encode an integer label vector.

        Parameters
        ----------
        vec : ndarray, shape (n,)
            Input labels.

        Returns
        -------
        values : ndarray
            Unique values in run order.
        lengths : ndarray
            Run lengths for each value.
        starts : ndarray
            Start indices of each run in the original vector.
        """

        if vec.size == 0:
            return np.array([]), np.array([]), np.array([])
        change = np.flatnonzero(np.diff(vec, prepend=vec[0] - 1))
        lengths = np.diff(np.append(change, vec.size))
        return vec[change], lengths, change

    @staticmethod
    def _kmed_1d(x: np.ndarray, k: int = 3, it_max: int = 30):
        """
        1D k-medians clustering using median updates and L1 assignment.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Scalar series to cluster.
        k : int
            Number of clusters.
        it_max : int
            Maximum number of update iterations.

        Returns
        -------
        labels : ndarray, shape (n,)
            Cluster labels in 0..k-1.
        centers : ndarray, shape (k,)
            Median centers after convergence.
        """

        cent = np.quantile(x, np.linspace(0.1, 0.9, k))
        for _ in range(it_max):
            lab = np.abs(x[:, None] - cent[None, :]).argmin(1)
            new_cent = np.array([np.median(x[lab == j]) for j in range(k)])
            if np.allclose(new_cent, cent, atol=1e-6):
                break
            # for empty centroids
            for j in range(k):
                if np.isnan(new_cent[j]):
                    new_cent[j] = cent[j] + (0.01 * np.std(x) or 1e-3)
            cent = new_cent
        return lab, cent

    
    @staticmethod
    def _effective_mu0(mu, sigma, delta):
        """
        Compute the first-coordinate mean corrected for skewness.

        This returns mu[0] plus the standard skew t shift term based on
        sigma and delta, falling back to mu[0] if the solve fails.

        Parameters
        ----------
        mu : ndarray, shape (p,)
            Location vector.
        sigma : ndarray, shape (p,p)
            Covariance matrix.
        delta : ndarray, shape (p,)
            Skewness vector.

        Returns
        -------
        float
            Effective mean on coordinate 0.
        """
        from scipy.linalg import solve

        try:
            q = solve(sigma, delta, assume_a='pos')
            shift = np.sqrt(sigma[0, 0]) * delta[0] / np.sqrt(1 + delta @ q)
        except Exception:
            shift = 0.0
        return mu[0] + shift * np.sqrt(2 / np.pi)


    
    def _smooth_and_seed_labels(self, X, *, med_win: int):
        """
        Smooth the first feature with a median filter and seed 3 labels.

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Observation matrix.
        med_win : int
            Median filter window size (will be forced to odd).

        Returns
        -------
        labels : ndarray, shape (n,)
            Seed labels mapped to {0=pos, 1=flat, 2=neg}.
        x0_sm : ndarray, shape (n,)
            Smoothed first coordinate used for seeding.
        """        
        from scipy.signal import medfilt
        x0 = X[:, 0]
        if (med_win & 1) == 0:
            med_win += 1
        x0_sm = medfilt(x0, med_win)

        labels_raw, centers = self._kmed_1d(x0_sm, k=3)
        
        idx_sort = np.argsort(centers)
        lut = np.empty(3, dtype=np.int32)
        lut[idx_sort[2]] = 0  # higher mean
        lut[idx_sort[1]] = 1  # in between
        lut[idx_sort[0]] = 2  # lower mean
        labels = lut[labels_raw.astype(np.int32)]
        return labels, x0_sm


    def _merge_short_segments(self, labels: np.ndarray, *, min_seg: int):
        """
        Merge segments shorter than a threshold into the nearest longer neighbor.

        Parameters
        ----------
        labels : ndarray, shape (n,)
            Current label sequence.
        min_seg : int
            Minimum allowed segment length.

        Returns
        -------
        labels : ndarray, shape (n,)
            Updated labels after merges (in place semantics).
        """
        vals, lens, starts = self._rle(labels)
        for v, L, s in zip(vals, lens, starts):
            if L >= min_seg:
                continue
            left = np.where(starts < s)[0]
            right = np.where(starts > s)[0]
            cand = []
            if left.size:
                cand.append(left[-1])
            if right.size:
                cand.append(right[0])
            tgt_idx = max(cand, key=lambda i: lens[i])
            labels[s:s+L] = vals[tgt_idx]
        return labels

    def _ensure_three_states_present(self, labels: np.ndarray, x0_sm: np.ndarray, p: int):
        """
        Ensure all three states appear; if not, enforce a 1/3 split by order.

        Parameters
        ----------
        labels : ndarray, shape (n,)
            Current label sequence.
        x0_sm : ndarray, shape (n,)
            Smoothed first coordinate for ranking.
        p : int
            Feature dimension, used as a minimal size guard.

        Returns
        -------
        labels : ndarray, shape (n,)
            Possibly replaced labels guaranteeing 3 states.
        """
        for st in (0, 1, 2):
            if (labels == st).sum() < p + 5:
                order = np.argsort(x0_sm)
                n = len(labels); chunk = n // 3
                newlab = np.zeros_like(labels)
                newlab[order[:chunk]] = 2    # lower mean
                newlab[order[chunk:2*chunk]] = 1  # in between
                newlab[order[2*chunk:]] = 0  # higher mean
                return newlab
        return labels

    def _estimate_mskt_params_per_state(self, X, labels, *, trim_q: float):
        """
        Estimate per-state skew t parameters via robust trimming and single-comp fits.

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Data matrix.
        labels : ndarray, shape (n,)
            State labels used to partition X.
        trim_q : float
            Quantile threshold for robust trimming by squared norm.

        Returns
        -------
        means : ndarray, shape (3, p)
        covs  : ndarray, shape (3, p, p)
        delt  : ndarray, shape (3, p)
        dof   : ndarray, shape (3,)
            Estimated parameters for each of the three states.
        """
        n, p = X.shape
        g = 3
        means = np.zeros((g, p))
        covs  = np.zeros((g, p, p))
        delt  = np.zeros((g, p))
        dof   = np.full(g, 6.0)

        
        for st in range(g):
            #1st step
            blk = X[labels == st]
            if blk.size == 0:
                # fallback 
                _log.info( "Fallback in first step of _estimate_mskt_params_per_state")
                means[st] = 0.0
                covs[st]  = np.eye(p)
                delt[st]  = 0.0
                dof[st]   = 8.0
                continue

            med = np.median(blk, axis=0)
            mad = np.median(np.abs(blk - med), axis=0) + 1e-9

            z = (blk - med) / mad
            # norm^2
            sq = np.einsum('ij,ij->i', z, z)

            # quantile through partition (O(n))
            k = int(max(0, min(len(sq) - 1, np.floor(trim_q * (len(sq) - 1)))))
            thr = np.partition(sq, k)[k]
            keep = sq <= thr
            trimmed = blk[keep]
            
            #2nd step
            try:
                pars = native.init_emmix_singlecomp(trimmed)
                means[st] = pars["mu"]
                covs[st]  = pars["sigma"]
                delt[st]  = pars["delta"]
                dof[st]   = pars["dof"]
            except Exception:
                _log.info( "Fallback in second step of _estimate_mskt_params_per_state")
                means[st] = trimmed.mean(0)
                covs[st]  = np.cov(trimmed, rowvar=False) if trimmed.shape[0] > 1 else np.eye(p)
                delt[st]  = 0.0
                dof[st]   = 8.0

            covs[st], delt[st], _ = native._safe_sigma_delta(covs[st], delt[st])

        return means, covs, delt, dof



    def _reorder_by_effective_mu0(self, means, covs, delt, dof, labels):
        """
        Reorder states by decreasing effective mean of coordinate 0.

        Parameters
        ----------
        means, covs, delt, dof
            Per-state parameters.
        labels : ndarray, shape (n,)
            Current labels to be permuted consistently.

        Returns
        -------
        means, covs, delt, dof : reordered parameter arrays
        labels : ndarray, shape (n,)
            Labels remapped to new order.
        eff_ordered : ndarray, shape (3,)
            Effective means in the new order.
        """
        eff = np.empty(3, dtype=np.float64)
        for i in range(3):
            Sigma = covs[i]
            delta = delt[i]
            try:
                c, lower = cho_factor(Sigma, check_finite=False)
                q = cho_solve((c, lower), delta, check_finite=False)
                shift = np.sqrt(Sigma[0, 0]) * delta[0] / np.sqrt(1.0 + delta @ q)
            except Exception:
                shift = 0.0
            eff[i] = means[i, 0] * 1.0 + shift * np.sqrt(2.0 / np.pi)

        order = np.argsort(eff)[::-1]  # high mean, in between, lower mean
        means = means[order]
        covs  = covs[order]
        delt  = delt[order]
        dof   = dof[order]

        # map labels
        lut = np.empty(3, dtype=np.int32)
        lut[order] = np.arange(3, dtype=np.int32)
        labels = lut[labels]

        return means, covs, delt, dof, labels.astype(np.int32), eff[order]


    def _estimate_discrete_from_labels(self, labels):
        """
        Estimate start probabilities and transition matrix from a label path.

        Parameters
        ----------
        labels : ndarray, shape (n,)
            Discrete labels.

        Returns
        -------
        startprob : ndarray, shape (g,)
            Initial state distribution estimated from labels[0].
        transmat : ndarray, shape (g, g)
            Row-stochastic transition matrix estimated by counts.
        """
        g = 3
        startprob = np.eye(g, dtype=float)[labels[0]]
        # transitions with bincount
        prev = labels[:-1]
        next_ = labels[1:]
        idx = prev * g + next_
        counts = np.bincount(idx, minlength=g*g).reshape(g, g)
        A = counts.astype(float)
        A += 1e-3
        A /= A.sum(1, keepdims=True)
        return startprob, A


    def _stash_emission_params(self, means, covs, delt, dof):
        """
        Store emission parameters into the model and keep immutable copies.

        This method updates means_, covars_, delta_, dof_ and their *_init_
        counterparts so that initial values can be inspected later.
        """
        self.means_init_, self.means_   = means.copy(), means.copy()
        self.covars_init_, self.covars_ = covs.copy(),  covs.copy()
        self.delta_init_, self.delta_   = delt.copy(),  delt.copy()
        self.dof_init_,   self.dof_     = dof.copy(),   dof.copy()


    def _single_start_from_emmix(self, X):
        """
        Initialize a single-component skew t fit using the native EMMIX hook.

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Data matrix.

        Returns
        -------
        mu0 : ndarray, shape (p,)
        Sigma0 : ndarray, shape (p, p)
        delta0 : ndarray, shape (p,)
        nu0 : float
            One-component parameter estimates.
        """
        covmap = {"full": 3}
        ncov   = covmap[self.covariance_type]
        base   = native.init_emmix_singlecomp(X, ncov=ncov, maxloop=20)
        return base["mu"], base["sigma"], base["delta"], base["dof"]

    #TODO : Remove delta and nu or perturb them
    def _duplicate_and_perturb(self, mu0, delta0, nu0, *, g, rng):
        """
        Duplicate single-component parameters across g states and optionally perturb.

        Parameters
        ----------
        mu0, delta0, nu0
            Base parameters.
        g : int
            Number of states.
        rng : RandomState-like
            Source of randomness for small mean perturbations.

        Returns
        -------
        means : ndarray, shape (g, p)
        delta : ndarray, shape (g, p)
        dof   : ndarray, shape (g,)
        """

        means = np.tile(mu0, (g, 1))
        delta = np.tile(delta0, (g, 1))
        dof   = np.full(g, nu0, dtype=np.float64)
        if g > 1:
            means += 1e-4 * rng.randn(*means.shape)
        return means, delta, dof

    def _covars_from_sigma0(self, Sigma0, *, g):
        """
        Convert a single full covariance matrix into hmmlearn internal format.

        Parameters
        ----------
        Sigma0 : ndarray, shape (p, p)
            Base full covariance.
        g : int
            Number of states.

        Returns
        -------
        covars : ndarray
            Covariance array in hmmlearn layout for the configured type.
        """
        if self.covariance_type == "full":
            Sigma_single = Sigma0.copy()
        else:
            raise ValueError("covariance_type not allowed")
        return _utils.distribute_covar_matrix_to_match_covariance_type(
            Sigma_single, covariance_type=self.covariance_type, n_components=g
        ).copy()


    
    def _init(self, X, lengths=None):
        """
        Entry point for model initialization.

        If n_components == 1, build a single-component start.
        Otherwise initialize discrete parameters via BaseHMM and
        seed emissions and labels with the robust segment procedure.
        """
        super()._init(X, lengths)
        g = self.n_components
        if g == 1:
            return self._init_single(X)
        self._init_robust_segments(X)


    def _init_robust_segments(self, X: np.ndarray, *, med_win: int = 51,
                          min_seg: int = 180, trim_q: float = 0.98, verbose:bool =False ) -> None:
        """
        Robust multi-stage initializer for a 3 state skew t HMM.

        Steps
        -----
        1) Smooth first feature and seed 3 labels by 1D clustering.
        2) Merge short segments.
        3) Ensure all three states appear.
        4) Fit per-state skew t parameters with trimming.
        5) Reorder states by effective mean on coordinate 0.
        6) Estimate start and transition probabilities from labels.
        7) Stash emission parameters and freeze labels for a few EM iters.

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Data matrix.
        med_win : int
            Median filter window for smoothing.
        min_seg : int
            Minimum segment length after merging.
        trim_q : float
            Trimming quantile for robust fits.
        verbose : bool
            for debugging
        """
        if verbose is None:
            verbose = bool(getattr(self, "verbose", False))

        labels, x0_sm = self._smooth_and_seed_labels(X, med_win=med_win)
        labels = self._merge_short_segments(labels, min_seg=min_seg)
        labels = self._ensure_three_states_present(labels, x0_sm, p=X.shape[1])

        means, covs, delt, dof = self._estimate_mskt_params_per_state(X, labels, trim_q=trim_q)
        means, covs, delt, dof, labels, _ = self._reorder_by_effective_mu0(
            means, covs, delt, dof, labels
        )

        startprob, A = self._estimate_discrete_from_labels(labels)
        self.startprob_ = startprob
        self.startprob_init_ = startprob.copy()
        self.transmat_ = A
        self.transmat_init_ = A.copy()

        self._stash_emission_params(means, covs, delt, dof)

        # frozen labels - first two steps of EM the emission parameters are frozen
        self._frozen_labels_ = {"labels": labels.astype(np.int32), "remain": 2}



   
   

    
    def _init_single(self, X, lengths=None):
        """
        Initialize a single-state skew t HMM.

        Uses a one-component EMMIX fit to get (mu, Sigma, delta, nu), then
        builds hmmlearn-format covariances and sets model fields.
        """

        if not (self._needs_init("m", "means_") or self._needs_init("c", "covars_")):
            return

        X = np.asarray(X, dtype=np.float64, order="C")
        g = self.n_components
        mu0, Sigma0, delta0, nu0 = self._single_start_from_emmix(X)

        rng = (check_random_state(self.random_state)
               if self.random_state is not None else np.random)
        means, delta, dof = self._duplicate_and_perturb(mu0, delta0, nu0, g=g, rng=rng)
        covars = self._covars_from_sigma0(Sigma0, g=g)

        self.means_ = means
        self.delta_ = delta
        self.dof_   = dof
        self._covars_ = covars

    
    def _do_mstep(self, stats):
        """
        Maximization step for skew t emissions and discrete parameters.

        For g > 1, update startprob_ and transmat_ via the base class.
        Then update emission parameters with a closed form M step using
        sufficient statistics (including latent moments) produced in E step.
        Degrees of freedom are updated when effective counts are sufficient.
        """
        # 0) \pi , A  
        if self.n_components > 1:
            super()._do_mstep(stats)

        g, p = self.n_components, self.n_features

        # 1) concatenate
        X = np.asfortranarray(np.concatenate(stats["X_blocks"], 0))
        tau = np.asfortranarray(np.concatenate(stats["tau_blocks"], 0))
        ev = np.asfortranarray(np.concatenate(stats["ev_blocks"], 0))
        ez1v = np.asfortranarray(np.concatenate(stats["ez1_blocks"], 0))
        ez2v = np.asfortranarray(np.concatenate(stats["ez2_blocks"], 0))

        sumtau = np.asfortranarray(stats["post"], dtype="float64")
        sumvt = np.asfortranarray(stats["sumvt"], dtype="float64")
        sumzt = np.asfortranarray(stats["sumzt"], dtype="float64")
        sumlnv = np.asfortranarray(stats["sumlnv"], dtype="float64")

        # 2) current parameters for Fortran
        mu_f = np.asfortranarray(self.means_.T, dtype="float64")  # (p,g)
        delta_f = np.asfortranarray(self.delta_.T, dtype="float64")  # (p,g)
        sigma_f = np.asfortranarray(
            _utils.covars_to_full(self._covars_, "full",
                                  n_components=g, n_features=p
                                  ).transpose(1, 2, 0), "float64")  # (p,p,g)

        covmap = {"full": 3}
        ncov   = covmap[self.covariance_type]

        # 3) M-step 
        mu_f, sigma_f, delta_f = native.mstepmst(
            X=X, tau=tau, ev=ev, ez1v=ez1v, ez2v=ez2v,
            sumtau=sumtau, sumvt=sumvt, sumzt=sumzt,
            mu=mu_f, sigma=sigma_f, delta=delta_f,
            ncov=ncov,
        )

        # 3-bis) only SPD safeguard
        self._sigma_fixed = False
        for k in range(g):
            Sigmak = sigma_f[:, :, k]
            try:
                np.linalg.cholesky(Sigmak)  # ok, no fix
            except np.linalg.LinAlgError:  # not SPD
                w, V = np.linalg.eigh(Sigmak)
                eps = 1e-8 * np.trace(Sigmak) / p
                w = np.maximum(w, eps)
                sigma_f[:, :, k] = (V * w) @ V.T
                self._sigma_fixed = True

        if self._sigma_fixed and _log.isEnabledFor(logging.INFO):
            _log.info("Sigma corrected (SPD jitter) at step %s",
                      getattr(self.monitor_, "iter", "?"))

        # 4) nu
        if np.all(sumtau > p + 2):
            self.dof_ = native.getdof(
                sumtau=sumtau, sumlnv=sumlnv,
                dof=np.asfortranarray(self.dof_, dtype="float64"),
            )

        # 5) copy
        self.means_ = mu_f.T
        self.delta_ = delta_f.T
        self._covars_ = sigma_f.transpose(2, 0, 1)




    
    def _check(self):
        super()._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

    def _needs_sufficient_statistics_for_mean(self):
        return 'm' in self.params

    def _needs_sufficient_statistics_for_covars(self):
        return 'c' in self.params

    def _needs_sufficient_statistics_for_skewness(self):
        return 'k' in self.params

    def _needs_sufficient_statistics_for_degrees_f(self):
        return 'v' in self.params

    # TODO : substitute static methods
    def _generate_sample_from_state(self):
        raise NotImplementedError