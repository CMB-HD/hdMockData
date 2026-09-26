# Forecasting Data for CMB-HD

This repository contains the forecasting data for CMB-HD, including:
- Lensed and delensed CMB $TT$, $TE$, $EE$, $BB$ and CMB lensing $\kappa\kappa$ power spectra,
- Noise curves for the spectra listed above, with and without including residual extragalaxtic foregrounds,
- Covariance matrices for the spectra listed above,
- The binning file used to bin the spectra and covariance matrices,
- The fiducial set of CAMB or CLASS parameters used in the theory calculations.

It also includes Python functions that can be used to access the data.

If you use any of the data, please cite (see "Data versions" below):
- For version `v1.2` (the default, latest version):
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


# CAMB and CLASS parameter files

We provide YAML files (in the `hd_mock_data/data/theory` directory) containing the fiducial set of CMB-HD CAMB parameters for each data version, and the corresponding CLASS parameters starting from version `v1.2`. These files include cosmological, accuracy, and any additional parameters that may be passed to CAMB or CLASS; dictionaries of the parameters may be loaded in using the `camb_settings` or `class_settings` methods of the `hd_data.HDMockData` class. 

## Note about CAMB versions

Our fiducial set of CAMB parameters includes a parameter named `lens_output_margin` starting in CAMB version 2.0.0, or named `lens_margin` in previous versions. When you access the CAMB parameters using the `camb_settings` or `camb_settings_fname` method of `hd_data.HDMockData`, we attempt to import CAMB in order to determine which CAMB version you are using, and return the appropriate parameter dictionary or file for that version. If CAMB cannot be imported, we default to the newer `lens_output_margin` name.


## CLASS modifications required to use the CLASS parameters provided here

**CLASS must be modified** in order to use the CMB-HD fiducial accuracy settings, and in order to vary the effective number of relativistic species (e.g., in Fisher forecasts or MCMC runs). These modifications must be made *before* compiling and installing CLASS (e.g., via `make` or `pip install`). 

The instructions are given in Appendix A of Cheslog et. al. (2026) and repeated below. We will use `$CLASS_DIR` to refer to the path to the cloned CLASS repository (named `class_public` by default); e.g., the directory that contains the CLASS "readme" and `explanatory.ini` files.

- In `$CLASS_DIR/source/lensing.c` ([line 124](https://github.com/lesgourg/class_public/blob/v3.3.4/source/lensing.c#L124) in version 3.3.4, or [lines 130-131](https://github.com/lesgourg/class_public/blob/v3.4.0/source/lensing.c#L130) in version 3.4.0), you *must* update the type of `num_mu` and `index_mu` to be `long long`, and update the type of `icount` to be `unsigned long long`. After these modifications, the relevant lines in the file should be:
    ```c
    
    unsigned long long icount;
    long long num_mu , index_mu;
    
    
    ```
  This is necessary in order to use `l_max_scalars` above approximately 14,000 when `accurate_lensing=1`.

- In `$CLASS_DIR/source/input.c`, we (strongly) recommend that you comment out the following line ([line 2470](https://github.com/lesgourg/class_public/blob/v3.3.4/source/input.c#L2470) in version 3.3.4 or [line 2548](https://github.com/lesgourg/class_public/blob/v3.4.0/source/input.c#L2548) in version 3.4.0):
    ```c
    
    /*
    class_test(pba->Omega0_ur<0,errmsg,"You cannot set the density of ultra-relativistic relics (dark radiation/neutrinos) to negative values. You might have input a total Neff smaller than what your massive neutrinos require minimally (around 1.02 * N_ncdm * deg_ncdm).");
    */
    
    
    ```
  This is necessary in order to vary `Neff` below approximately 3.0396 (or, equivalently, the `N_ur` parameter below zero) with three massive neutrinos (`N_ncdm=3`). You *must* comment out this line if you are using CLASS for Fisher forecasts, MCMC runs, etc., but it is *not* required in order to use the fiducial set of parameters provided in `hdMockData`.
  
- In order to use the `sBBN file` provided here (`hd_mock_data/data/theory/PRIMAT_Yp_DH_ErrorMC_2021_CLASS.dat`, which is a copy of the corresponding file provided by [CAMB](https://github.com/cmbant/CAMB/blob/master/camb/PRIMAT_Yp_DH_ErrorMC_2021.dat), formatted for CLASS), you *must* place a copy of that file in the `$CLASS_DIR/external/bbn/` directory (please do not *move* it from `hdMockData`!). You may copy it over yourself, or follow these instructions:

  First, make sure that `hd_mock_data` has been correctly installed by running the command `python -c "from hd_mock_data import hd_data` ; if nothing happens, you may proceed; otherwise, follow the instructions above to install `hd_mock_data`. 
  
  Then, run the following two commands from within your `$CLASS_DIR`:
  
  ```bash
  CLASS_SBBN_FILE=$(python -c "from hd_mock_data import hd_data; print(hd_data.class_sbbn_file())")
  
  ```
  
  ```bash
  cp $CLASS_SBBN_FILE external/bbn/
  ```

