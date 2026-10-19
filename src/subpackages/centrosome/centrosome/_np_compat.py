import numpy as np

# Check NumPy version once
NP_MAJOR_VERSION = int(np.__version__.split(".")[0])

# Set up aliases based on version
if NP_MAJOR_VERSION >= 2:
    # NumPy 2.x aliases
    np_product = np.prod
    np_cumproduct = np.cumprod
    np_NaN = np.nan
    np_Inf = np.inf
else:
    # NumPy 1.x aliases
    np_product = np.product
    np_cumproduct = np.cumproduct
    np_NaN = np.NaN
    np_Inf = np.Inf 