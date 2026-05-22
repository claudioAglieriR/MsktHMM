
import warnings

import numpy as np
from scipy import stats


def mardia_test(X, *, subsample: int = 2000, random_state: int = 0):
    """Mardia's multivariate skewness and kurtosis normality test.

    Used by the initializer to decide whether a regime needs a finite skew-t
    degrees-of-freedom nu (heavy tails) or can keep nu pinned at the cap. The
    kurtosis component is the tail-heaviness signal that matters for nu; the
    skewness component is reported for diagnostics (skew is the job of delta).

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Observations for one state.
    subsample : int
        Cap on rows used; the skewness statistic needs an n-by-n Mahalanobis
        matrix, so very large states are subsampled to bound cost.
    random_state : int
        Seed for the subsampling.

    Returns
    -------
    dict or None
        None if the sample is too small to test. Otherwise keys: skew_stat,
        skew_df, skew_p, kurtosis, kurt_z, kurt_p, n, p. P-values below the
        chosen alpha indicate departure from normality.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return None
    n, p = X.shape
    if n < max(20, 2 * p + 2):
        return None
    if n > subsample:
        rng = np.random.default_rng(random_state)
        X = X[rng.choice(n, subsample, replace=False)]
        n = subsample

    Xc = X - X.mean(axis=0)
    S = np.cov(Xc, rowvar=False)
    Sinv = np.linalg.pinv(S)
    M = Xc @ Sinv @ Xc.T                      # (n, n) Mahalanobis cross-products

    # Mardia skewness: A ~ chi^2 with p(p+1)(p+2)/6 dof under normality
    A = n * float((M ** 3).mean()) / 6.0
    df = p * (p + 1) * (p + 2) / 6.0
    skew_p = float(stats.chi2.sf(A, df))

    # Mardia kurtosis: standardized to z ~ N(0, 1) under normality
    d = np.diag(M)                            # squared Mahalanobis distances
    b2p = float((d ** 2).mean())
    z = (b2p - p * (p + 2)) / np.sqrt(8.0 * p * (p + 2) / n)
    kurt_p = float(2.0 * stats.norm.sf(abs(z)))

    return {"skew_stat": A, "skew_df": df, "skew_p": skew_p,
            "kurtosis": b2p, "kurt_z": float(z), "kurt_p": kurt_p,
            "n": int(n), "p": int(p)}

# ── k-medians defaults ────────────────────────────────────────────────────────
_KMED_N_CLUSTERS = 3    # default number of clusters
_KMED_MAX_ITER   = 30   # maximum update iterations


def covars_to_full(covars, covariance_type, *, n_components=None, n_features=None):
    """
    Convert the internal HMM covariance storage to a (g, p, p) tensor.
    Only used for the call to native.ddmix(). Only "full" covariance is supported.
    """
    if covariance_type != "full":
        raise ValueError(
            f"covariance_type must be ‘full’; got {covariance_type!r}"
        )
    return covars
    
    
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



def _kmed_1d(x: np.ndarray, k: int = _KMED_N_CLUSTERS, it_max: int = _KMED_MAX_ITER):
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
