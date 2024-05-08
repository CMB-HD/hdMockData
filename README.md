# Forecasting Data for CMB-HD

This repository contains the forecasting data for CMB-HD, including:
- Lensed and delensed CMB $TT$, $TE$, $EE$, $BB$ and CMB lensing $\kappa\kappa$ power spectra,
- Noise curves for the spectra listed above, with and without including residual extragalaxtic foregrounds,
- Covariance matrices for the spectra listed above,
- The binning file used to bin the spectra and covariance matrices.

It also includes Python functions that can be used to access the data.

If you use any of the data, please cite (see "Data versions" below):
- MacInnis & Sehgal (2024) for version `v1.1` (this is the default version).
- [MacInnis, Sehgal, and Rothermel (2023)](https://arxiv.org/abs/2309.03021) for version `v1.0`.

# Installation

There are no requirements to access the data files themselves; they are stored in the `hd_mock_data/data` directory. 

To easily load the files using the Python functions in `hd_mock_data/hd_data.py`, you must have Python (version >=3) and [NumPy](https://numpy.org/) installed. Then, simply clone this repository and install with `pip`:

```
git clone https://github.com/CMB-HD/hdMockData.git
cd hdMockData
pip install . --user
```

# Useage

The functions to load the data are located in the `HDMockData` class of `hd_mock_data/hd_data.py`. 

We label each file by a version number (see the "Data Versions" section below). The `HDMockData` class takes in a single variable, the `version`. By default, `version = 'latest'`, which automatically uses the latest version of the data.

For example: to load the CMB noise curves into a Python dictionary named `noise`, your Python code would be:

```
from hd_mock_data import hd_data
hd_data_lib = hd_data.HDMockData()
noise = hd_data_lib.cmb_noise_spectra()
```

# Data versions

- `v1.1`: Currently the latest. Contains the data used in MacInnis & Sehgal (2024).
- `v1.0`: Contains the data used in [MacInnis, Sehgal, and Rothermel (2023)](https://arxiv.org/abs/2309.03021). This version should be used only to reproduce the results of that work.

Note that there may not be a new version for each data product, if that data product was not updated. The Python code will automatically find the correct version of each file.


