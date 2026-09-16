# Forecasting Data for CMB-HD

This repository contains the forecasting data for CMB-HD, including:
- Lensed and delensed CMB $TT$, $TE$, $EE$, $BB$ and CMB lensing $\kappa\kappa$ power spectra,
- Noise curves for the spectra listed above, with and without including residual extragalaxtic foregrounds,
- Covariance matrices for the spectra listed above,
- The binning file used to bin the spectra and covariance matrices.

It also includes Python functions that can be used to access the data.

If you use any of the data, please cite (see "Data versions" below):
- For version `'v1.2` (the default, latest version):
  - [MacInnis et. al. (2026)](https://arxiv.org/abs/2609.16128) if you use the CMB-HD noise spectra, foreground spectra, or covariance matrices.
  - [Cheslog et. al. (2026)](https://arxiv.org/abs/YYYY.YYYYY) (to be submitted) if you use the theory power spectra or CAMB/CLASS parameter settings.
- [MacInnis & Sehgal (2024)](https://arxiv.org/abs/2405.12220) for version `v1.1`.
- [MacInnis, Sehgal, and Rothermel (2023)](https://arxiv.org/abs/2309.03021) for version `v1.0`.


# Installation

There are no requirements to access the data files themselves; they are stored in the `hd_mock_data/data` directory. 

To easily load the files using the Python functions in `hd_mock_data/hd_data.py`, you must have Python (version >=3), [NumPy](https://numpy.org/), and [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation) installed. Then, simply clone this repository and install with `pip`:

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

See the `example.ipynb` Jupyter notebook for more detailed examples.


# Data versions

- `v1.2`: Currently the latest. Contains new simulation-based noise curves and corresponding covariance matrices from [MacInnis et. al. (2026)](https://arxiv.org/abs/2609.16128), along with the CAMB/CLASS accuracy settings from [Cheslog et. al. (2026)](https://arxiv.org/abs/YYYY.YYYYY). The changes relative to the previous version are:
  - Simulation-based estimates of CMB-HD residual extragalactic foregrounds at 90 and 150 GHz, and the resulting CMB temperature and CMB lensing noise power spectra, which are used to calculate new lensed and delensed covariance matrices.
    - Note that the temperature maximum multipole for this data is 20,000 (as in `v1.0`).
  - An alternative set of lensing noise curves and covariance matrices calculated using polarization-only ($EE$ and $EB$) lensing estimators.
  - Updated CAMB accuracy settings, new CLASS accuracy settings, and corresponding CMB and CMB lensing theory power spectra.
    - Note that we do not provide delensed CLASS CMB theory power spectra, because CLASS does not provide it.
    - To use the CLASS accuracy settings, please follow the CLASS modifications specified in Appendix A of [Cheslog et. al. (2026)](https://arxiv.org/abs/YYYY.YYYYY).
- `v1.1`: Contains the data used in [MacInnis & Sehgal (2024)](https://arxiv.org/abs/2405.12220). The changes relative to the previous version are:
  - Extended the temperature maximum multipole to 40,000.
  - Added the late-time kSZ, in addition to the reionization kSZ.
  - Removed an off-diagonal term in the covariance matrix (see footnote 4 in paper).
- `v1.0`: Contains the data used in [MacInnis, Sehgal, and Rothermel (2023)](https://arxiv.org/abs/2309.03021). This version should be used only to reproduce the results of that work.

Note that there may not be a new version for each data product, if that data product was not updated. The Python code will automatically find the correct version of each file.


