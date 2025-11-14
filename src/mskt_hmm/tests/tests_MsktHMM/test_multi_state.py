import os
import itertools
from pathlib import Path
import numpy as np
import pytest
import logging
from datetime import datetime

from src.mskt_hmm.mskt_hmm import MsktHMM
from src.mskt_hmm import native  
from joblib import Parallel, delayed

# NB: The data must be unzipped from the data_test folder to conduct this test
INPUT_CSV = Path(r"C:\opt\workspace\python\MsktHMM\src\mskt_hmm\tests\tests_MsktHMM\data_test\test_multi_state\parameters_distributions.csv")
DATA_DIR  = Path(r"C:\opt\workspace\python\MsktHMM\src\mskt_hmm\tests\tests_MsktHMM\data_test\test_multi_state\simulation_data_multi_state")
LOG_DIR = Path(r"C:\opt\workspace\python\MsktHMM\src\mskt_hmm\tests\tests_MsktHMM\log")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("tests.mskt_hmm")

POSITIVE_FIT_IDS = [101, 104, 107, 110]
NULL_FIT_IDS     = [102, 105, 108, 111]
NEGATIVE_FIT_IDS = [103, 106, 109, 112]
LIST_FIT_IDS     = [list(c) for c in itertools.product(POSITIVE_FIT_IDS, NULL_FIT_IDS, NEGATIVE_FIT_IDS)]

N_STATES        = 3
BLOCK_SIZES     = [50, 125, 250, 500, 1_000, 1_500, 2_000, 2_500, 3_000]
PER_SRC_TRAIN   = 2400
PER_SRC_TEST    = 2600
MIN_TRAIN_TOTAL = PER_SRC_TRAIN * N_STATES
MIN_TEST_TOTAL  = PER_SRC_TEST  * N_STATES
WINDOW_NEAR     = max(1, int(round(0.12 * min(BLOCK_SIZES))))  

N_TRIPLETS= 15

THRESHOLD_PERCENT_TRAINING = 0.3  # = 0.3%, i.e. max percentage error
THRESHOLD_PERCENT_TESTING = 1  # = 1%, i.e. max percentage error

def _data_available() -> bool:
    """Return True if required CSV inputs and at least one probe data file exist.
    This is a quick guard to skip tests when the synthetic MST data is not present
    on disk. It checks a single probe file under DATA_DIR to avoid scanning the
    entire directory.
    Returns
    -------
    bool
        True if INPUT_CSV exists and a probe file in DATA_DIR exists, False otherwise.
    """

    if not INPUT_CSV.exists():
        return False
    probe = DATA_DIR / f"mst_data_fit_{POSITIVE_FIT_IDS[0]}.csv"
    return probe.exists()

skip_no_fortran = pytest.mark.skipif(
    native.LIB is None, reason="libemmixskew not uploaded."
)
skip_no_data = pytest.mark.skipif(
    not _data_available(), reason=f"Data not found in in {DATA_DIR}"
)

# --------------------- HELPER: build sequences, mapping, near flags ----------
def build_sequence_multi(arrays, block_sizes, rng, *, min_total):
    """Concatenate random blocks from multiple source arrays into one sequence.
    At each step a source index is sampled (not repeating the previous one when
    possible), then a block length is sampled from `block_sizes`, and that block
    is appended. This continues until the concatenated length reaches at least
    `min_total`. A label vector indicates which source produced each row.
    Parameters
    ----------
    arrays : list of ndarray
        List of arrays with shape (n_i, p). Each array is a pool for one state.
    block_sizes : sequence of int
        Candidate block lengths to draw at each step.
    rng : numpy.random.Generator
        Random generator used for sampling sources and block sizes.
    min_total : int, keyword-only
        Minimum total number of rows required in the output.
    Returns
    -------
    X : ndarray
        Concatenated data of shape (N, p) with N >= min_total.
    y : ndarray
        Integer labels of shape (N,) with values in [0, len(arrays)-1].

    """

    n_src   = len(arrays)
    lengths = [len(a) for a in arrays]
    idx     = [0] * n_src
    prev    = None
    tot     = 0
    Xc, yc  = [], []
    while tot < min_total:
        cand = [s for s in range(n_src) if idx[s] < lengths[s] and s != prev]
        if not cand:
            cand = [s for s in range(n_src) if idx[s] < lengths[s]]
            if not cand:
                break
        s       = rng.choice(cand)
        blk_len = rng.choice(block_sizes)
        start   = idx[s]; end = min(start + blk_len, lengths[s])
        Xc.append(arrays[s][start:end])
        yc.append(np.full(end - start, s, dtype=int))
        idx[s] = end; tot += end - start; prev = s
    if tot < min_total:
        raise ValueError(f"Insufficient data: {tot} < {min_total}")
    return np.concatenate(Xc), np.concatenate(yc)

def best_state_mapping(y_true, y_pred, n_states=3):
    """
    Map predicted HMM state labels to the ground-truth label space via permutation.

    HMM state labels are only identifiable up to permutation, so a decoder may
    assign different numeric IDs to the same underlying states (e.g., predicted
    state 0 corresponds to true state 2). This function searches over all
    permutations of labels and returns the mapping that maximizes the number of
    matches between y_pred and y_true.

    This is purely a relabeling step: it does not alter the timing or count of
    state occurrences, so it cannot artificially improve results beyond resolving
    label-identity ambiguity.

    Parameters
    ----------
    y_true : array-like of int, shape (N,)
        Ground-truth state labels.
    y_pred : array-like of int, shape (N,)
        Predicted state labels to be permuted.
    n_states : int, default=3
        Number of distinct states.

    Returns
    -------
    mapping : dict
        A dictionary {pred_label: mapped_label} representing the best permutation.
    """

    best_map = {i: i for i in range(n_states)}
    best_hit = -1
    for perm in itertools.permutations(range(n_states)):
        mapped = np.vectorize(lambda s: perm[s])(y_pred)
        hits   = np.sum(mapped == y_true)
        if hits > best_hit:
            best_hit = hits
            best_map = {k: perm[k] for k in range(n_states)}
    return best_map

def mark_near_boundaries(labels, win):
    """
    Mark samples that are within a window of any state-change boundary.
    Given a 1D array of labels, every index within `win` steps (inclusive) of a
    change point is flagged True. All others are False.
    Parameters
    ----------
    labels : array-like of int, shape (N,)
        Sequence of state labels.
    win : int
        Half window size around each boundary. Must be >= 0.

    Returns
    -------
    flags : ndarray of bool, shape (N,)
        True where near a boundary, False elsewhere.
    """

    flags  = np.zeros_like(labels, dtype=bool)
    bounds = np.where(np.diff(labels) != 0)[0] + 1
    for b in bounds:
        flags[max(0, b - win): b + win + 1] = True
    return flags



def pytest_generate_tests(metafunc):
    """
    Parametrize the 'fit_triplet' fixture with N_TRIPLETS randomly selected triplets.

    Uses OS entropy by default so selections differ across runs. If the
    environment variable PYTEST_MSKT_SEED is set, selections become reproducible.
    """
    if "fit_triplet" in metafunc.fixturenames:
        seed_env = os.environ.get("PYTEST_MSKT_SEED")
        rng = np.random.default_rng(None if seed_env is None else int(seed_env))

        idxs = rng.choice(len(LIST_FIT_IDS), size=N_TRIPLETS, replace=False)
        chosen = [LIST_FIT_IDS[i] for i in idxs]
        metafunc.parametrize("fit_triplet", chosen, ids=[f"fit-{i}" for i in idxs])


def test_decode_on_training_and_predict_stream_last_point_with_global_threshold(fit_triplet):
    """
    End-to-end test for MsktHMM training and streaming last-point prediction.
    Workflow:
    1) Load three sources identified by `fit_triplet` for training and test streams.
    2) Build a training sequence and a test sequence by interleaving random blocks.
    3) Fit a 3-state MsktHMM on the training sequence.
    4) Map the Viterbi path to ground-truth states to avoid "naming errors".
    5) Assert the training error rate is below a global percentage threshold.
    6) Evaluate a streaming scenario: for each growing prefix, predict only the
    last state without lookahead, then compute error rate on the test stream.
    The test also logs near vs far errors relative to state-change boundaries.

    Parameters
    ----------
    fit_triplet : list[int]
        Three fit IDs: positive, null, negative.

    Asserts
    -------
    Training error percentage is <= THRESHOLD_PERCENT.

    Side Effects
    -----------
    Logs metrics and error breakdowns via the module logger.
    """

    import pandas as pd
    logger.info("start test for FIT_ID=%s", fit_triplet)

    arrays_fit, arrays_pred = [], []
    for fid in fit_triplet:
        df = pd.read_csv(DATA_DIR / f"mst_data_fit_{fid}.csv", header=None)
        X  = df.to_numpy(float)
        arrays_fit.append(X[:PER_SRC_TRAIN])
        arrays_pred.append(X[PER_SRC_TRAIN: PER_SRC_TRAIN + PER_SRC_TEST])

    rng = np.random.default_rng(42)
    X_fit, y_fit_true   = build_sequence_multi(arrays_fit,  BLOCK_SIZES, rng, min_total=MIN_TRAIN_TOTAL)
    X_pred, y_pred_true = build_sequence_multi(arrays_pred, BLOCK_SIZES, rng, min_total=MIN_TEST_TOTAL)
    logger.info("sequences ready: X_fit=%s, X_pred=%s", X_fit.shape, X_pred.shape)

    model = MsktHMM(
        n_components   = N_STATES,
        n_iter         = 8000,
        tol            = 1e-6,
        params         = "stmckv",
        init_params    = "stmc",
        random_state   = 10_000,
        algorithm      = "map",
    )
    logger.info("start fit()...")

    model.fit(X_fit)
    logger.info("Fit completed. Last LL=%.3f, iter=%d",
                    model.monitor_.history[-1], model.monitor_.iter)
    _, z_path = model.decode(X_fit, [len(X_fit)], algorithm="viterbi")
    mapping   = best_state_mapping(y_fit_true, z_path, n_states=N_STATES)
    z_mapped  = np.vectorize(mapping.get)(z_path)

    err_mask  = (z_mapped != y_fit_true)
    near_flags= mark_near_boundaries(y_fit_true, WINDOW_NEAR)

    tot_err   = int(err_mask.sum())
    near_err  = int(np.sum(err_mask & near_flags))
    far_err   = int(np.sum(err_mask & ~near_flags))
    perc_err_training = tot_err / len(X_fit) * 100.0
    assert perc_err_training <= THRESHOLD_PERCENT_TRAINING, (
        f"FIT_ID {fit_triplet}: training %error {perc_err_training:.6f}% "
        f"> threshold {THRESHOLD_PERCENT_TRAINING:.4f}% (max baseline + 0.1)"
    )

    # --- TEST-STREAM: predict only last point, otherwise we would have lookahead
    full_seq       = np.vstack([X_fit, X_pred])
    n_train        = len(X_fit)
    prefix_lengths = np.arange(n_train + 1, n_train + len(X_pred) + 1)

    def _predict_last_state_mapped(L: int) -> int:
        z_seq = model.predict(full_seq[:L], [L])
        last_state = z_seq[-1]
        return mapping.get(last_state, -1)

    pred_stream = np.array(
        Parallel(n_jobs=-1, backend="loky")(delayed(_predict_last_state_mapped)(int(L))
                                            for L in prefix_lengths)
    )

    err_mask_stream = (pred_stream != y_pred_true)
    near_flags_pr   = mark_near_boundaries(y_pred_true, WINDOW_NEAR)

    total_err  = int(err_mask_stream.sum())
    near_err_s = int(np.sum(err_mask_stream & near_flags_pr))
    far_err_s  = int(np.sum(err_mask_stream & ~near_flags_pr))
    err_rate_stream = total_err / len(y_pred_true)
    assert (100.0 * err_rate_stream) <= THRESHOLD_PERCENT_TESTING, (
        f"FIT_ID {fit_triplet}: test-stream %error {100.0 * err_rate_stream:.6f}% "
        f"> threshold {THRESHOLD_PERCENT_TESTING:.4f}% (max baseline + 0.1)"
    )

    logger.info("Test-stream: tot_err=%d (%.2f%%), near=%d, far=%d",
                    total_err, 100*err_rate_stream, near_err_s, far_err_s)

    logger.info(f"\n Training errors: {tot_err} / {len(X_fit)} ({perc_err_training:.6f}%)")
    logger.info(f"   near state-change : {near_err} | far from state-change: {far_err}\n")
    logger.info(f"\n Test-stream errors: {total_err} / {len(y_pred_true)} ({err_rate_stream:.2%})")
    logger.info(f"    near state-change : {near_err_s} | far from state-change: {far_err_s}\n\n\n\n\n")
