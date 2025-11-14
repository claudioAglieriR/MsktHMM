import numpy as np

import pytest

from sklearn.datasets import make_spd_matrix
from sklearn.utils import check_random_state

from hmmlearn.utils import normalize
from hmmlearn.base import DECODER_ALGORITHMS

# Make NumPy complain about underflows/overflows etc.
np.seterr(all="warn")


