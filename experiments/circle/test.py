import numpy as np
from torsk.models.initialize import sparse_nzpr_esn_reservoir

dim = 6
spectral_radius = 2.
nonzeros_per_row = 3
dtype = np.float32

matrix = sparse_nzpr_esn_reservoir(dim, spectral_radius, nonzeros_per_row, dtype)

