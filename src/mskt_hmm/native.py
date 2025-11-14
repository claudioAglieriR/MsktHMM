# native.py
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import cffi

_log = logging.getLogger(__name__)
ffi = cffi.FFI()

ffi.cdef(
    r"""
    /* ---------- initfit_ già presente ---------- */
    void initfit_(double *y, const int *n, const int *p, const int *g,
                  const int *ncov, const int *ndist,
                  double *pro,    double *mu,     double *sigma,
                  double *dof,    double *delta,
                  double *tau, double *ev, double *elnv,
                  double *ez1v, double *ez2v,
                  double *sumtau, double *sumvt,
                  double *sumzt,  double *sumlnv,
                  double *ewy, double *ewz, double *ewyy,
                  double *loglik,
                  int    *clust,
                  int    *error,
                  const int *maxloop);

    /* ---------- ddmix2  (due varianti) ---------- */
    void ddmix2_(double *x, const int *n, const int *p, const int *g,
                 const int *ndist,
                 const double *mu, const double *sigma,
                 const double *dof, const double *delta,
                 double *den, int *error);

    void ddmix2 (double *x, const int *n, const int *p, const int *g,
                 const int *ndist,
                 const double *mu, const double *sigma,
                 const double *dof, const double *delta,
                 double *den, int *error);
    /* ---- denmst2 : E-step momenti skew-t ------------------------- */
    void denmst2_(double *x, const int *n, const int *p, const int *g,
                  const double *mu, const double *sigma,
                  const double *dof, const double *delta,
                  double *tau, double *ev, double *elogv,
                  double *ez1v, double *ez2v,
                  int *error, int *method);
    /* =====  mstepmst : M-step closed-form per MST  ===== */
    void mstepmst_(double *y, const int *n, const int *p, const int *g,
                   const int *ncov,
                   double *tau,  double *ev,
                   double *ez1v, double *ez2v,
                   double *sumtau, double *sumvt, double *sumzt,
                   double *mu, double *sigma, double *delta);

    /* =====  getdof : aggiornamento ν  ===== */
    void getdof_(int *pn, int *pg,
                 double *sumtau, double *sumlnv,
                 double *dof, double *b);
    """
)


_DISTMAP = {"mvn": 1, "mvt": 2, "msn": 3, "mst": 4}

def _sym(name: str):
    """
    Return a symbol from the loaded shared library by trying common Fortran/C
    name mangling variants. Tries single and double underscores and uppercase
    forms. Raises AttributeError if the symbol cannot be found.

    Parameters
    ----------
    name : str
        Base symbol name without trailing underscores.

    Returns
    -------
    object
        The callable symbol object from the shared library.

    Raises
    ------
    AttributeError
        If no variant of the symbol is found in the shared library.
    """

    variants = [
        f"{name}_", f"{name}__", name,
        f"{name}_".upper(), f"{name}__".upper(), name.upper(),
    ]
    for cand in variants:
        try:
            return getattr(LIB, cand)
        except AttributeError:
            continue
    raise AttributeError(f"symbol {name} not found in shared library")

def _dlopen():
    """
    Load the EMMIXskew shared library.

    Search order:
      1) same directory as this module (native.py)
      2) fixed subdir:  <module_dir>/EMMIXskew_dll
      3) any subdir matching: <module_dir>/mskt-hmm-natives-*
    Looks for typical library basenames: libemmixskew / EMMIXskew
    and platform extensions (.dll / .so / .dylib).
    """
    here = Path(__file__).resolve().parent

    import sys
    if sys.platform.startswith("win"):
        exts = [".dll"]
    elif sys.platform == "darwin":
        exts = [".dylib", ".so"]
    else:
        exts = [".so"]

    fixed_subdir = here / "EMMIXskew_dll"
    dynamic_subdirs = sorted(here.glob("mskt-hmm-natives-*"))
    search_dirs = [here]
    if fixed_subdir.exists():
        search_dirs.append(fixed_subdir)
    search_dirs.extend([d for d in dynamic_subdirs if d.is_dir() and d != fixed_subdir])

    base_names = ("libemmixskew", "EMMIXskew")

    tried = []
    for d in search_dirs:
        for base in base_names:
            for ext in exts:
                cand = d / f"{base}{ext}"
                tried.append(str(cand))
                if cand.exists():
                    try:
                        return ffi.dlopen(str(cand))
                    except OSError:
                        # continua a provare altri candidati
                        pass

        # fallback
        for pat in (f"{base_names[0]}.*", f"{base_names[1]}.*"):
            for so in d.glob(pat):
                tried.append(str(so))
                try:
                    return ffi.dlopen(str(so))
                except OSError:
                    pass

    _log.warning(
        "EMMIXskew shared library not found. Tried: %s",
        "; ".join(tried) if tried else "(no candidates)"
    )
    return None


LIB = _dlopen()







def _force_spd(M: np.ndarray, eps_scale: float = 1e-5) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest symmetric positive definite
    matrix using an eigenvalue floor. If the input already passes a Cholesky
    factorization, it is returned unchanged.

    Parameters
    ----------
    M : ndarray, shape (p, p)
        Input matrix, not necessarily SPD.
    eps_scale : float
        Fraction of trace used to floor eigenvalues.

    Returns
    -------
    ndarray
        A symmetric positive definite matrix.
    """

    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = 0.5 * (M + M.T)                 

    try:                                # is it SPD?
        np.linalg.cholesky(M)
        return M                        
    except np.linalg.LinAlgError:
        pass

    
    w, V = np.linalg.eigh(M)
    eps  = eps_scale * np.trace(M) / len(M)
    w[w < eps] = eps
    return (V * w) @ V.T                



def _is_spd(A: np.ndarray) -> bool:
    """
    Quick SPD test via Cholesky factorization.
    Parameters
    ----------
    A : ndarray, shape (p, p)
        Matrix to test.

    Returns
    -------
    bool
        True if A is symmetric positive definite, False otherwise.
    """

    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def _safe_sigma_delta(
    sigma: np.ndarray,
    delta: np.ndarray,
    verbose: bool = False,
    *,
    tol_quad: float = 0.99
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Validate and minimally repair the pair (sigma, delta) for the unrestricted
    multivariate skew-t parameterization. Ensures sigma is SPD and that
    delta' inv(sigma) delta is below a threshold. If needed, repairs by
    projecting sigma to SPD and scaling delta.

    Parameters
    ----------
    sigma : ndarray, shape (p, p)
        Covariance matrix.
    delta : ndarray, shape (p,)
        Skewness vector.
    verbose : bool
        If True, prints basic diagnostics when a repair is applied.
    tol_quad : float
        Upper bound for the quadratic form delta' inv(sigma) delta.

    Returns
    -------
    sigma_new : ndarray
        Possibly repaired SPD covariance.
    delta_new : ndarray
        Possibly scaled skewness vector.
    fixed : bool
        True if any repair was applied, False otherwise.
    """

    fixed = False

    if not _is_spd(sigma):
        old_sigma=sigma
        sigma = _force_spd(sigma)
        fixed = True
        if verbose:
            print(f"FORCING COVARIANCE MATRICE SDP - OLD COVARIANCE MATRIX = {old_sigma}")
            print(f"NEW COVARIANCE MATRIX = {sigma}")


    quad = float(delta @ np.linalg.solve(sigma, delta))
    if not np.isfinite(quad) or quad >= tol_quad:
        old_quad=quad
        scale = np.sqrt(1.05 * max(quad, tol_quad))
        delta = delta / scale
        fixed = True
        if verbose:
            print(f"PROBLEM WITH QUAD - OLD QUAD = {old_quad}")
            print(f"NEW QUAD = {float(delta @ np.linalg.solve(sigma, delta))}")

    return sigma, delta, fixed



def _call_initfit(
    X: np.ndarray,
    *, g: int = 1, ncov: int = 3, maxloop: int = 20,
    clust_in: np.ndarray | None = None,
):
    """
    Low-level wrapper around the Fortran routine initfit_. Calls the library
    to produce initial parameters for a skew family mixture, returning flat
    arrays in Fortran order along with labels and log-likelihood.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix.
    g : int
        Number of components.
    ncov : int
        Covariance structure code expected by the Fortran routine.
    maxloop : int
        Maximum iterations for the initializer.
    clust_in : ndarray or None, shape (n,)
        Optional 1-based initial labels.

    Returns
    -------
    mu : ndarray, shape (p*g,)
        Means flattened in Fortran order.
    sigma : ndarray, shape (Ltri*g,)
        Packed lower-triangular covariance entries per component.
    delta : ndarray, shape (p*g,)
        Skewness vectors flattened.
    dof : ndarray, shape (g,)
        Degrees of freedom per component.
    clust : ndarray, shape (n,)
        1-based labels from the initializer.
    loglik : float
        Initial log-likelihood value.
    err : int
        Fortran error code, 0 if success.

    Raises
    ------
    RuntimeError
        If the shared library is not loaded.
    """

    if LIB is None:
        raise RuntimeError("libemmixskew not uploaded")

    # ----- basic dims -------------------------------------------------
    Xf = np.asfortranarray(X, dtype="float64")
    n, p   = Xf.shape
    ndist  = 4
    Ltri   = p * (p + 1) // 2

    # ----- allocate outputs -------------------------------------------
    pro   = np.full(g, 1.0/g, order="F")
    mu    = np.zeros(p * g,    order="F")
    sigma = np.zeros(Ltri * g, order="F")        # vech lower-col
    dof   = np.full(g, 10.0,   order="F")
    delta = np.zeros(p * g,    order="F")

    # ----- labels (1-based) ------------------------------------------
    if clust_in is None:
        clust = np.ones(n, dtype="int32", order="F")
    else:
        clust = np.asfortranarray(clust_in, dtype="int32")
        if clust.min() < 1 or clust.max() > g or clust.shape != (n,):
            raise ValueError("clust_in must be 1…g e shape=(n,)")

    error  = np.zeros(1, dtype="int32")

    # ----- workspace --------------------------------------------------
    wk = lambda m: np.zeros(m, dtype="float64", order="F")
    tau = wk(n*g); ev = wk(n*g); elnv = wk(n*g)
    ez1v = wk(n*g); ez2v = wk(n*g)
    sumtau = wk(g); sumvt = wk(g); sumzt = wk(g); sumlnv = wk(g)
    ewy = wk(p*g);  ewz = wk(p*g);  ewyy = wk(Ltri*g)
    loglik = wk(1)

    # ----- call Fortran ----------------------------------------------
    LIB.initfit_(
        ffi.cast("double *", Xf.ctypes.data),
        ffi.new("int *", n),  ffi.new("int *", p),  ffi.new("int *", g),
        ffi.new("int *", ncov), ffi.new("int *", ndist),
        *(ffi.cast("double *", a.ctypes.data) for a in
          (pro, mu, sigma, dof, delta,
           tau, ev, elnv, ez1v, ez2v,
           sumtau, sumvt, sumzt, sumlnv,
           ewy, ewz, ewyy, loglik)),
        ffi.cast("int *", clust.ctypes.data),
        ffi.cast("int *", error.ctypes.data),
        ffi.new("int *", maxloop),
    )

    return mu, sigma, delta, dof, clust, float(loglik[0]), int(error[0])




def init_emmix_multicomp(
    X: np.ndarray, *,
    g: int,
    ncov: int = 3,
    maxloop: int = 50,
    clust_in: np.ndarray | None = None,
    verbose: bool = False
):
    """
    High-level initializer for multi-component skew-t models. Calls the
    Fortran initfit routine, rebuilds full covariance matrices from packed
    form, and applies minimal SPD and skewness safeguards.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix.
    g : int
        Number of components.
    ncov : int
        Covariance structure code expected by the Fortran routine.
    maxloop : int
        Maximum iterations for the initializer.
    clust_in : ndarray or None, shape (n,)
        Optional 1-based initial labels.
    verbose : bool
        If True, prints basic diagnostics when repairs are applied.

    Returns
    -------
    dict
        Keys: "mu" (g, p), "sigma" (g, p, p), "delta" (g, p),
        "dof" (g,), "labels" (n,) zero-based, and "loglik" (float).

    Raises
    ------
    RuntimeError
        If initfit returns a nonzero error code.
    """

    n, p = X.shape
    (mu_flat, sigma_tri, delta_flat, dof,
     clust, loglik, err) = _call_initfit(
        X, g=g, ncov=ncov, maxloop=maxloop,
        clust_in=clust_in)

    if err != 0:
        raise RuntimeError(f"initfit_ error code {err}")

    Ltri = p * (p + 1) // 2
    sigma_tri = sigma_tri.reshape(g, Ltri, order="F")

    # ----------------------------------------------------------------------
    # init_emmix_multicomp – rebuild \Sigma
    # ----------------------------------------------------------------------
    sigma_full = np.zeros((g, p, p))
    for k in range(g):
        idx = 0
        for j in range(p):                 # lower-col vech
            for i in range(j + 1):
                sigma_full[k, i, j] = sigma_full[k, j, i] = sigma_tri[k, idx]
                idx += 1

        sigma_k = sigma_full[k]
        delta_k = delta_flat[k*p:(k+1)*p]

        if (not _is_spd(sigma_k) or
            float(delta_k @ np.linalg.solve(sigma_k, delta_k)) >= 0.99):
            if verbose:
                print(f"\n\n\nPROBLEM WITH STATE # {k} WITH delta OR covariance matrix")
            sigma_k, delta_k, _ = _safe_sigma_delta(sigma_k, delta_k)
            sigma_full[k] = sigma_k
            delta_flat[k*p:(k+1)*p] = delta_k



    return {
        "mu":     mu_flat.reshape(g, p, order="F"),
        "sigma":  sigma_full,
        "delta":  delta_flat.reshape(g, p, order="F"),
        "dof":    dof.copy(),
        "labels": clust.astype(int) - 1,   # 0-based
        "loglik": loglik,
    }



def getdof(
    *,
    sumtau: np.ndarray,
    sumlnv: np.ndarray,
    dof:    np.ndarray,
    upper: float = 200.0,
):
    """
    Wrapper for the Fortran getdof_ routine that updates the degrees of
    freedom for skew-t components. Cleans NaN and inf values in sufficient
    statistics and ensures a minimal effective sample size.

    Parameters
    ----------
    sumtau : ndarray, shape (g,)
        Component responsibilities summed over observations.
    sumlnv : ndarray, shape (g,)
        Sum of E[log lambda | y] per component.
    dof : ndarray, shape (g,)
        Current degrees of freedom, updated in place.
    upper : float
        Upper cap used by the Fortran routine.

    Returns
    -------
    ndarray
        The updated dof array.

    Raises
    ------
    RuntimeError
        If the shared library is not loaded.
    """

    if LIB is None:
        raise RuntimeError("libemmixskew not uploaded")

    # Replace bad values
    sumtau = np.nan_to_num(sumtau, nan=0.0, posinf=0.0, neginf=0.0)
    sumlnv = np.nan_to_num(sumlnv, nan=0.0, posinf=0.0, neginf=0.0)

    n_eff = int(max(1, round(float(sumtau.sum()))))

    _sym("getdof_")(
        ffi.new("int *", n_eff),
        ffi.new("int *", dof.size),
        ffi.cast("double *", sumtau.ctypes.data),
        ffi.cast("double *", sumlnv.ctypes.data),
        ffi.cast("double *", dof.ctypes.data),
        ffi.new("double *", upper),
    )
    return dof







def _call_initfit_single(X: np.ndarray, *, ncov: int = 3, maxloop: int = 20):
    """
    Low-level call to the Fortran initfit_ for the single-component case
    (g = 1). Returns flat arrays in Fortran order and the error code.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix.
    ncov : int
        Covariance structure code.
    maxloop : int
        Maximum iterations for the initializer.

    Returns
    -------
    mu : ndarray, shape (p,)
    sigma : ndarray, shape (p*p,)
        Packed as a flat Fortran-ordered p x p matrix.
    delta : ndarray, shape (p,)
    dof : float
    err : int
        Fortran error code, 0 if success.
    """
    if LIB is None:
        return None

    Xf = np.asfortranarray(X, dtype="float64")
    n, p = Xf.shape
    g, ndist = 1, 4                      # 1-component skew-t

    # ---- buffer output: **zero-init** come in R --------------------
    pro   = np.zeros(g,        dtype="float64", order="F")   
    mu    = np.zeros(p * g,    dtype="float64", order="F")   
    sigma = np.zeros(p * p * g,dtype="float64", order="F")   
    dof   = np.zeros(g,        dtype="float64", order="F")   
    delta = np.zeros(p * g,    dtype="float64", order="F")   

    clust = np.ones(n, dtype="int32")                        
    error = np.zeros(1, dtype="int32")                       


    wk = lambda n_el: np.zeros(n_el, dtype="float64", order="F")
    tau     = wk(n * g); ev = wk(n * g); elnv = wk(n * g)
    ez1v    = wk(n * g); ez2v = wk(n * g)
    sumtau  = wk(g);     sumvt = wk(g);    sumzt = wk(g); sumlnv = wk(g)
    ewy     = wk(p * g); ewz   = wk(p * g); ewyy = wk(p * p * g)
    loglik  = wk(1)


    LIB.initfit_(
        ffi.cast("double *", Xf.ctypes.data),
        ffi.new("int *", n),  ffi.new("int *", p),  ffi.new("int *", g),
        ffi.new("int *", ncov), ffi.new("int *", ndist),
        *(ffi.cast("double *", arr.ctypes.data) for arr in
          (pro, mu, sigma, dof, delta,
           tau, ev, elnv, ez1v, ez2v,
           sumtau, sumvt, sumzt, sumlnv,
           ewy, ewz, ewyy,
           loglik)),
        ffi.cast("int *",  clust.ctypes.data),
        ffi.cast("int *",  error.ctypes.data),
        ffi.new("int *", maxloop),
    )

    return mu, sigma, delta, dof[0], int(error[0])


# -------------------------------------------------------------------------
def init_emmix_singlecomp(X: np.ndarray, *, ncov: int = 3, maxloop: int = 20):
    """
    High-level single-component initializer for an unrestricted multivariate
    skew-t distribution. Wraps _call_initfit_single and reshapes outputs.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix.
    ncov : int
        Covariance structure code.
    maxloop : int
        Maximum iterations for the initializer.

    Returns
    -------
    dict
        Keys: "mu" (p,), "sigma" (p, p), "delta" (p,), "dof" (float).

    Raises
    ------
    RuntimeError
        If the shared library is not loaded or initfit returns an error.
    """
    out = _call_initfit_single(X, ncov=ncov, maxloop=maxloop)
    if out is None:
        raise RuntimeError("libemmixskew non caricata")

    mu, sigma_flat, delta, dof, err = out
    if err != 0:
        raise RuntimeError(f"initfit_ error code {err}")

    p = mu.size
    return {
        "mu":    mu,                                   # (p,)
        "sigma": sigma_flat.reshape(p, p, order="F"),  # (p,p)
        "delta": delta,                                # (p,)
        "dof":   dof,                                  # float
    }





def ddmix(
    X: np.ndarray,
    *,
    distr: str,
    mu:    np.ndarray,
    sigma: np.ndarray,
    dof:   np.ndarray | list | tuple,
    delta: np.ndarray,
    g:     int,
) -> np.ndarray:
    """
    Compute log densities for a set of mixture components using the Fortran
    ddmix2 kernel. Supports mvn, mvt, msn, and mst based on the 'distr'
    selector. All parameter arrays must be Fortran-contiguous float64.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix.
    distr : str
        One of 'mvn', 'mvt', 'msn', or 'mst'.
    mu : ndarray, shape (p*g,)
        Means flattened in Fortran order.
    sigma : ndarray, shape (p*p*g,)
        Covariances flattened in Fortran order.
    dof : array-like, shape (g,)
        Degrees of freedom per component, used for t and skew-t.
    delta : ndarray, shape (p*g,)
        Skewness vectors flattened, used for skew-n and skew-t.
    g : int
        Number of components.

    Returns
    -------
    ndarray, shape (n, g)
        Log density values for each observation and component.

    Raises
    ------
    RuntimeError
        If the shared library is not loaded or the kernel returns an error.
    ValueError
        If input shapes are inconsistent.
    """

    if LIB is None:
        raise RuntimeError("libemmixskew not uploaded")

    distr = distr.lower()
    if distr not in _DISTMAP:
        raise ValueError(f"unknown distr '{distr}' (expected mvn/mvt/msn/mst)")
    ndist = _DISTMAP[distr]

    # ---- (0)   shape & cast -------------------------------------------------
    Xf = np.asfortranarray(X, dtype="float64")
    n, p = Xf.shape
    g = int(g)

    mu     = np.asfortranarray(mu,    dtype="float64")
    sigma  = np.asfortranarray(sigma, dtype="float64")
    delta  = np.asfortranarray(delta, dtype="float64")
    dof    = np.asfortranarray(dof,   dtype="float64")

    if mu.size != p * g:
        raise ValueError(f"mu length should be p*g = {p*g}, got {mu.size}")
    if sigma.size != p * p * g:
        raise ValueError(f"sigma length should be p*p*g = {p*p*g}, got {sigma.size}")
    if delta.size != p * g:
        raise ValueError(f"delta length should be p*g = {p*g}, got {delta.size}")
    if dof.size != g:
        raise ValueError(f"dof length should be g = {g}, got {dof.size}")

    # ---- (1)   buffer output -----------------------------------------------
    den   = np.empty(n * g, dtype="float64", order="F")
    error = np.empty(1,     dtype="int32")

    # ---- (2)   call ddmix2 --------------------------------------------------
    _sym("ddmix2")(
        ffi.cast("double *", Xf.ctypes.data),
        ffi.new("int *", n),   ffi.new("int *", p),  ffi.new("int *", g),
        ffi.new("int *", ndist),
        ffi.cast("double *", mu.ctypes.data),
        ffi.cast("double *", sigma.ctypes.data),
        ffi.cast("double *", dof.ctypes.data),
        ffi.cast("double *", delta.ctypes.data),
        ffi.cast("double *", den.ctypes.data),
        ffi.cast("int *",    error.ctypes.data),
    )

    errcode = int(error[0])
    if errcode != 0:
        raise RuntimeError(f'ddmix2 error code {errcode}')

    # ---- (3)   reshape & return --------------------------------------------
    return den.reshape((n, g), order="F")


def denmst(
    *,
    X:     np.ndarray,
    distr: str,
    mu:    np.ndarray,
    sigma: np.ndarray,
    dof:   np.ndarray,
    delta: np.ndarray,
    g:     int,
):
    """
    Full E-step helper for the multivariate skew-t distribution. Calls the
    Fortran denmst2 kernel to compute log densities and conditional moments:
    E[lambda | y], E[|Z|*lambda | y], E[(|Z|*lambda)^2 | y], and E[log lambda | y].

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix.
    distr : str
        Must be 'mst'.
    mu : ndarray, shape (p*g,)
        Means flattened in Fortran order.
    sigma : ndarray, shape (p*p*g,)
        Covariances flattened in Fortran order.
    dof : ndarray, shape (g,)
        Degrees of freedom per component.
    delta : ndarray, shape (p*g,)
        Skewness vectors flattened.
    g : int
        Number of components.

    Returns
    -------
    logdens : ndarray, shape (n, g)
        Log density for each observation and component.
    ev : ndarray, shape (n, g)
        E[lambda | y].
    ez1v : ndarray, shape (n, g)
        E[|Z|*lambda | y].
    ez2v : ndarray, shape (n, g)
        E[(|Z|*lambda)^2 | y].
    elnv : ndarray, shape (n, g)
        E[log lambda | y].

    Raises
    ------
    RuntimeError
        If the shared library is not loaded or the kernel returns an error.
    ValueError
        If 'distr' is not 'mst'.
    """

    if LIB is None:
        raise RuntimeError("libemmixskew not uploaded")

    if distr.lower() != "mst":
        raise ValueError("denmst implemented only for 'mst' (skew-t)")

    # ---- (0)  cast a Fortran-order ----------------------------------
    Xf = np.asfortranarray(X, dtype="float64")
    n, p = Xf.shape
    g = int(g)

    mu     = np.asfortranarray(mu,    dtype="float64")
    sigma  = np.asfortranarray(sigma, dtype="float64")
    delta  = np.asfortranarray(delta, dtype="float64")
    dof    = np.asfortranarray(dof,   dtype="float64")

    # ---- (1)  buffer output -----------------------------------------
    tau   = np.empty(n * g, dtype="float64", order="F")   # log-dens
    ev    = np.empty(n * g, dtype="float64", order="F")
    elogv = np.empty(n * g, dtype="float64", order="F")
    ez1v  = np.empty(n * g, dtype="float64", order="F")
    ez2v  = np.empty(n * g, dtype="float64", order="F")
    error = np.empty(1,     dtype="int32")
    method= np.ones(g,      dtype="int32")  

    # ---- (2)  call denmst2 ------------------------------------------
    _sym("denmst2_")(
        ffi.cast("double *", Xf.ctypes.data),
        ffi.new("int *", n),   ffi.new("int *", p),  ffi.new("int *", g),
        ffi.cast("double *", mu.ctypes.data),
        ffi.cast("double *", sigma.ctypes.data),
        ffi.cast("double *", dof.ctypes.data),
        ffi.cast("double *", delta.ctypes.data),
        ffi.cast("double *", tau.ctypes.data),
        ffi.cast("double *", ev.ctypes.data),
        ffi.cast("double *", elogv.ctypes.data),
        ffi.cast("double *", ez1v.ctypes.data),
        ffi.cast("double *", ez2v.ctypes.data),
        ffi.cast("int *",    error.ctypes.data),
        ffi.cast("int *",    method.ctypes.data),
    )

    if int(error[0]) != 0:
        raise RuntimeError(f'denmst2 error code {int(error[0])}')

    # ---- (3)  reshape & return --------------------------------------
    shp = (n, g)
    logdens = tau.reshape(shp, order="F")
    ev      = ev .reshape(shp, order="F")
    ez1v    = ez1v.reshape(shp, order="F")
    ez2v    = ez2v.reshape(shp, order="F")
    elnv    = elogv.reshape(shp, order="F")

    return logdens, ev, ez1v, ez2v, elnv





def mstepmst(
    *,
    X, tau, ev, ez1v, ez2v,
    sumtau, sumvt, sumzt,
    mu, sigma, delta,
    ncov: int,
):
    """
    Closed-form M-step for the unrestricted multivariate skew-t distribution.
    Wraps the Fortran mstepmst_ kernel to update means, covariances, and
    skewness given sufficient statistics from the E-step.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Data matrix (Fortran contiguous).
    tau : ndarray, shape (n, g)
        Posterior responsibilities per component.
    ev : ndarray, shape (n, g)
        E[lambda | y].
    ez1v : ndarray, shape (n, g)
        E[|Z|*lambda | y].
    ez2v : ndarray, shape (n, g)
        E[(|Z|*lambda)^2 | y].
    sumtau : ndarray, shape (g,)
        Sum of responsibilities per component.
    sumvt : ndarray, shape (g,)
        Sum of E[lambda | y] per component.
    sumzt : ndarray, shape (g,)
        Sum of E[(|Z|*lambda)^2 | y] per component.
    mu : ndarray, shape (p, g)
        Current means, updated in place and returned.
    sigma : ndarray, shape (p, p, g)
        Current covariances, updated in place and returned.
    delta : ndarray, shape (p, g)
        Current skewness vectors, updated in place and returned.
    ncov : int
        Covariance structure code expected by the Fortran routine.

    Returns
    -------
    mu : ndarray, shape (p, g)
        Updated means.
    sigma : ndarray, shape (p, p, g)
        Updated covariances.
    delta : ndarray, shape (p, g)
        Updated skewness vectors.

    Raises
    ------
    RuntimeError
        If the shared library is not loaded.
    """

    if LIB is None:
        raise RuntimeError("libemmixskew not uploaded")

    n, p        = X.shape
    g           = int(sumtau.size)

    # buffer in/out ---------------------------------------------------
    _sym("mstepmst_")(
        ffi.cast("double *", X.ctypes.data),
        ffi.new("int *", n),  ffi.new("int *", p),  ffi.new("int *", g),
        ffi.new("int *", ncov),
        *(ffi.cast("double *", a.ctypes.data) for a in
          (tau, ev, ez1v, ez2v,
           sumtau, sumvt, sumzt,
           mu, sigma, delta))
    )
    return mu, sigma, delta
