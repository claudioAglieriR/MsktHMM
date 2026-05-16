
import warnings

import numpy as np

# ── k-medians defaults ────────────────────────────────────────────────────────
_KMED_N_CLUSTERS = 3    # default number of clusters
_KMED_MAX_ITER   = 30   # maximum update iterations


def covars_to_full(covars, covariance_type, *, n_components=None, n_features=None):
    """
    Convert the internal HMM covariance storage to a (g, p, p) tensor.
    Only used for the call to native.ddmix().
    """
    if covariance_type == "full":
        return covars
    if n_components is None or n_features is None:
        # Infer from covars’ shape – handles already-broadcasted cases.
        if covariance_type == "tied":
            n_features = covars.shape[0]
            n_components = 1
        elif covariance_type == "diag":
            n_components, n_features = covars.shape
        elif covariance_type == "spherical":
            n_components = covars.shape[0]
            n_features = 1
    if covariance_type == "tied":
        return np.tile(covars, (n_components, 1, 1))
    elif covariance_type == "diag":
        return np.array([np.diag(cv) for cv in covars], dtype=np.float64)
    elif covariance_type == "spherical":
        return np.array([np.eye(n_features) * s for s in covars],
                        dtype=np.float64)
    else:
        raise ValueError("unknown covariance_type")
    
    
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
