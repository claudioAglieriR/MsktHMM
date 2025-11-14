import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.mskt_hmm.mskt_hmm import MsktHMM   

# NB: The data must be unzipped from the data_test folder to conduct this test
HERE        = Path(__file__).resolve().parent
DATA_DIR    = HERE / "data_test" / "test_single_state_equivalence"

GT_CSV      = DATA_DIR / "parameters_distributions_single_state.csv"
R_CSV       = DATA_DIR / "EMMIX_fit.csv"
ALLOWED_CSV = DATA_DIR / "error_comparison_EMMIX_R.csv"
SIM_DIR     = DATA_DIR / "simulation_data"

ERROR_COLS = [
    "err_dof",
    *[f"err_mu_{i}"     for i in range(3)],
    *[f"err_delta_{i}"  for i in range(3)],
    *[f"err_Sigma_{i}{j}" for i in range(3) for j in range(3)],
]

THRESHOLD = 0.05          # acceptable error treshold
N_ITER    = 8_000
TOL       = 1e-6


gt_df       = pd.read_csv(GT_CSV)
r_df        = pd.read_csv(R_CSV).set_index("fit")          # R‐errors
allowed_df  = pd.read_csv(ALLOWED_CSV)                     # acceptable deltas
diff_col    = next(c for c in allowed_df.columns           
                   if c.lower() == "differences")


def rel_err(est, tru):
    """
    Compute elementwise relative error with a safe denominator.

    This helper returns abs(est - tru) / denom where denom is:
      - abs(tru) when tru != 0
      - 1.0 when tru == 0 (to avoid division by zero)

    Inputs can be scalars or numpy arrays; standard numpy broadcasting
    rules apply. 
    Parameters
    ----------
    est : array-like or float
        Estimated value(s).
    tru : array-like or float
        Ground-truth value(s).
    Returns
    -------
    np.ndarray
        Nonnegative array of relative errors with the broadcasted shape.
    """

    est = np.asarray(est, float)
    tru = np.asarray(tru, float)
    denom = np.where(tru != 0, np.abs(tru), 1.0)
    return np.abs(est - tru) / denom


@pytest.mark.parametrize("row", gt_df.to_dict("records"))
def test_mskt_hmm_against_r(row):
    """
    One-state uMST HMM vs EMMIX (R) equivalence test for a single fit id.

    For each ground-truth configuration (identified by 'fit'):
      1) Load the simulated dataset and the target parameters
         (mu, delta, Sigma, dof) from CSV files.
      2) Fit a 1-state MsktHMM with full covariance to the data
         using EM.
         The discrete HMM part is fixed to a single absorbing state
         (startprob = 1, transmat = [[1.0]]).
      3) Compute relative errors for all parameters and compare them
         against the errors produced by the reference R implementation
         stored in EMMIX_fit.csv.
      4) Accept if:
           - the HMM error is below a small global threshold, or
           - the HMM error is not worse than R, or
           - if worse, it is within the allowed per-metric delta
             listed in error_comparison_EMMIX_R.csv.
         Otherwise the test fails with a readable assertion message.

    Files expected under data_test/test_single_state_equivalence:
      - parameters_distributions_single_state.csv (ground truth)
      - EMMIX_fit.csv (reference R errors)
      - error_comparison_EMMIX_R.csv (allowed error deltas)
      - simulation_data/mst_data_fit_<fit>.csv (one per fit id)

    Notes
    -----
    - The model uses params="stmckv" so start, trans, mean, cov, skew,
      and degrees of freedom are all enabled during EM.
    - The Skew-t emission is the unrestricted multivariate version.
    - This test asserts parity at the level of parameter errors rather
      than exact equality of fitted parameters.
    """


    # ---------- Ground-truth parameters ----------------------------
    fit_id     = int(row["fit"])
    mu_true    = np.array([row[f"mu_true_{i}"]    for i in range(3)], dtype=float)
    delta_true = np.array([row[f"delta_true_{i}"] for i in range(3)], dtype=float)
    Sigma_true = np.array([row[f"Sigma_true_{i}{j}"]
                           for i in range(3) for j in range(3)], dtype=float).reshape(3, 3)
    dof_true   = float(row["dof_true"])

    # ---------- Simulated dataset ----------------------------------
    X = pd.read_csv(SIM_DIR / f"mst_data_fit_{fit_id}.csv",
                    header=None).to_numpy(float)

    # ---------- Fitting ------------------------------------------
    model = MsktHMM(
        n_components=1,
        n_iter=N_ITER,
        tol=TOL,
        params="stmckv",
        init_params="stmc",
        random_state=fit_id * 1_000,        # seed
    )
    model.startprob_ = 1
    model.transmat_  = np.array([[1.0]])
    model.fit(X)

    # ---------- Esimates and errors of In-House HMM, i.e. MsktHM 
    mu_hat    = model.means_[0]
    delta_hat = model.delta_[0]
    Sigma_hat = model.covars_[0]
    dof_hat   = model.dof_[0]

    err_hmm = {
        "err_dof":           float(rel_err(dof_hat, dof_true)),
        **{f"err_mu_{i}":    float(rel_err(mu_hat[i],    mu_true[i]))    for i in range(3)},
        **{f"err_delta_{i}": float(rel_err(delta_hat[i], delta_true[i])) for i in range(3)},
        **{f"err_Sigma_{i}{j}": float(rel_err(Sigma_hat[i, j], Sigma_true[i, j]))
            for i in range(3) for j in range(3)},
    }

    # ---------- Errors of original EMMIXSkew, generated by R+EMMIXSkew library  
    row_r = r_df.loc[fit_id]
    if isinstance(row_r, pd.DataFrame):
        row_r = row_r.iloc[0]

    err_r = {col: float(row_r[col]) for col in ERROR_COLS}

    allowed_diff = (allowed_df[allowed_df["fit"] == fit_id]
                    .set_index("errore")[diff_col]
                    .to_dict())

    # ---------- ASSERTIONS ----------------------------------------
    for col in ERROR_COLS:
        e_hmm, e_r = err_hmm[col], err_r[col]

        # criterion 0: small error
        if e_hmm <= THRESHOLD:
            continue

        # criterion 1: HMM better than R
        ok = e_hmm <= e_r

        # criteriom 2: if worse, not greater than admissible delta
        if not ok:
            prev = allowed_diff.get(col, np.inf)   
            ok = round(abs(e_hmm - e_r),4) <= round(abs(prev),4)

        assert ok, (
            f"fit={fit_id}  {col}:  "
            f"HMM={e_hmm:.4g}  R={e_r:.4g}  "
            f"delta={e_hmm - e_r:+.4g}  "
            f"(accepted ±{abs(allowed_diff.get(col, np.nan)):.4g})"
        )
