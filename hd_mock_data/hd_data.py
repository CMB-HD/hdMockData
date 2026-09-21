import os
import warnings
import numpy as np
import yaml


def binning_matrix(bin_edges, lmin=None, lmax=None, start_at_ell=2):
    """Create a (num_bins, num_ells) binning matrix, which will bin the
    values between `lmin` and `lmax` in a vector/matrix containing values
    for each multipole between the `start_at_ell` and `lmax` values. For
    example, for an array `c_ell` holding a power spectrum with a value at
    each multipole `ell` in the range [2, 5000], to bin only the values in
    the range [30, 3000], you would pass `lmin = 30`, `lmax = 3000`, and 
    `start_at_ell=2`.

    Parameters
    ----------
    bin_edges : array of int
        A one dimensional array holding the upper bin edge for each bin,
        except the first element, which is the lower bin edge of the
        first bin.
    lmin, lmax : int or None, default=None
        The minimum and maximum multipole values of the quantity to be
        binned, i.e. only values between `lmin` and `lmax` will be
        binned. If `lmin` is `None`, we use the first value in the
        `bin_edges` array; if `lmax` is `None`, we use the last value in
        the `bin_edges` array.
    start_at_ell : int, default=2
        The minimum multipole value in the quantity to be binned. This is
        typically either `0` or `2`.

    Returns
    -------
    binmat : array of float
        The two-dimensional binning matrix of shape (num_bins, num_ells).
    """
    lmin = int(lmin) if (lmin is not None) else int(bin_edges[0])
    lmax = int(lmax) if (lmax is not None) else int(bin_edges[-1])
    ell_min = int(start_at_ell)
    ells = np.arange(ell_min, lmax+1)
    nells = len(ells)
    # get upper and lower edges
    upper = bin_edges[1:].copy()
    lower = bin_edges[:-1].copy()
    # add one to all lower edges, except the first,
    # so each bin includes its lower and its upper edge
    lower[1:] += 1
    # trim between lmin and lmax
    loc = np.where((lower >= lmin) & (upper <= lmax))
    upper = upper[loc]
    lower = lower[loc]
    nbin = len(upper)
    # make binning matrix
    binmat = np.zeros((nbin, nells))
    for i, (bmin, bmax) in enumerate(zip(lower, upper)):
        loc = np.where((ells >= bmin) & (ells <= bmax))
        n = bmax - bmin + 1 # number of ells in this bin
        binmat[i][loc] = 1 / n
    return binmat


def load_from_file(fname, columns, skip_cols=[]):
    """Returns dict of columns loaded from a `.txt` file.

    Parameters
    ----------
    fname : str
        The filename to load from.
    columns : list of str
        The names, in order, of each column, which will also serve as the
        dict keys.
    skip_cols : list of str, default=[]
        The names of any columns that should not be included in the output
        dict.

    Returns
    -------
    data : dict of array_like
        A dictionary whose keys are the names in `columns` and values are
        one-dimensional arrays holding the data from the corresponding
        column in the file.
    """
    data = {}
    data_array = np.loadtxt(fname)
    for i, col in enumerate(columns):
        if col not in skip_cols:
            data[col] = data_array[:,i].copy()
    return data


def _use_v2_camb_names():
    """Try to determine if the installed CAMB version is >= 2.0.0

    Returns `True` if the CAMB version is >= 2.0.0, `False` for lower
    versions, or `None` if CAMB is not installed.
    """
    try:
        import camb
        # we only need the first number:
        vnum = int(camb.__version__.split('.')[0])
        use_v2_names = (vnum >= 2)
    except ImportError:
        use_v2_names = None
    return use_v2_names


def _camb_param_names_warning(file_or_dict):
    """Warn the user when the CAMB version cannot be determined,
    so the parameter names will only work with CAMB version 2.0.0 or
    higher.
    """
    msg = ("Unable to determine the CAMB version. The returned "
           f"{file_or_dict} will contain parameter names "
           "compatible with CAMB version 2.0.0 or higher. If you are "
           "using a lower CAMB version, pass `v2_camb_names=False`.")
    warnings.warn(msg, stacklevel=2)


class HDMockData:
    data_versions = ['v1.0', 'v1.1', 'v1.2']
    latest_version = data_versions[-1]
    
    def __init__(self, version='latest'):
        if 'late' in version.lower():
            self.version = self.latest_version
        else:
            self.check_version(version)
            self.version = version.lower()

        # keep track of versions for each kind of file:
        self.binning_versions = ['v1.0', 'v1.1']
        self.theo_versions = self.data_versions
        self.mcmc_bandpower_versions = self.data_versions
        self.fg_versions = self.data_versions
        self.cl_ksz_versions = ['v1.1']
        self.cmb_noise_versions = self.data_versions # includes FG in TT
        self.cmb_white_noise_versions = ['v1.0', 'v1.2'] # white noise only
        self.lensing_noise_versions = ['v1.0', 'v1.2']
        self.covmat_versions = self.data_versions # full 5 x 5 covmat, 30 < ell < 20k
        self.tt_covmat_versions = ['v1.1'] # diagonal TTxTT, 20k < ell < 40k
        self.nlkk_pol_versions = ['v1.2'] # polarization-only lensing reconstruction
        self.camb_theo_versions = self.data_versions # CAMB settings
        self.class_theo_versions = ['v1.2'] # CLASS settings / theory spectra 

        # multipoles:
        self.lmin = 30
        self.lmax = 20100
        self.Lmin = 30
        self.Lmax = 20100
        self.lmaxTT = 40000 if (self.version == 'v1.1') else self.lmax
        # currently, coadded white noise saved up to lmax = 40,000;
        # make this a variable, in case it gets updated in the future:
        self.cmb_white_noise_lmax = 40000
        # same as above for multipoles used to calculate lensing noise:
        self.nlkk_lmin = 30
        self.nlkk_Lmin = 30
        self.nlkk_lmax = 20100
        self.nlkk_Lmax = 20100
        # and for covmats:
        self.covmat_lmin = 30
        self.covmat_Lmin = 30
        self.covmat_lmax = 20100
        self.covmat_Lmax = 20100
        self.tt_covmat_lmin = 20100
        self.tt_covmat_lmax = 40000
        # lmax for theory and noise spectra:
        if self.version in ['v1.0', 'v1.1']:
            self.theo_lmax = self.lmaxTT
            self.noise_lmax = self.lmaxTT
        else:
            self.theo_lmax = 24000
            self.noise_lmax = 23900 # max. bin edge for sim-based noise

        self.fsky = 0.6
        if self.version == 'v1.2':
            self.fsky *= 0.985 # 1.5% of the sky is masked after FG cleaning
        self.ells = np.arange(self.lmaxTT + 1)
        self.theo_cols = ['ells', 'tt', 'te', 'ee', 'bb', 'kk']
        self.noise_cols = self.theo_cols[:-1]
        if self.version in ['v1.0', 'v1.1']:
            self.fg_cols = ['ells', 'ksz', 'tsz', 'cib', 'radio']
        else:
            self.fg_cols = ['ells', 'tsz', 'cib_radio']
        self.cmb_types = ['lensed', 'delensed', 'unlensed']
        self.freqs = ['f090', 'f150']
        self.noise_levels = {'f090': 0.7, 'f150': 0.8} # uK-arcmin
        self.beam_fwhm = {'f090': 0.42, 'f150': 0.25} # arcmin
        # use enhanced SO BB noise for ell < 1000:
        if self.version in ['v1.0', 'v1.1']: # prelim noise levels
            self.aso_noise_levels = {'f090': 3.5, 'f150': 3.8} # uK-arcmin
        else: # from table 1 of arXiv:2503.00636
            self.aso_noise_levels = {'f090': 3.8, 'f150': 4.1} # uK-arcmin
        self.aso_beam_fwhm = {'f090': 2.2, 'f150': 1.4} # arcmin

        # paths to files:
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
        self.data_path = lambda x: os.path.join(self.data_dir, x)
        self.binning_path = lambda x: os.path.join(self.data_path('binning'), x)
        self.theo_path = lambda x: os.path.join(self.data_path('theory'), x)
        self.cdm_theo_path = lambda x: os.path.join(self.theo_path('cdm'), x)
        self.cdm_baryons_theo_path = lambda x: os.path.join(self.theo_path('cdm_baryonic_feedback'), x)
        self.noise_path = lambda x: os.path.join(self.data_path('noise'), x)
        self.fg_path = lambda x: os.path.join(self.data_path('foregrounds'), x)
        self.covmat_path = lambda x: os.path.join(self.data_path('covariance_matrices'), x)
        self.class_sbbn_file = self.theo_path('PRIMAT_Yp_DH_ErrorMC_2021_CLASS.dat')

        # try to check which CAMB version is being used
        self._use_v2camb = _use_v2_camb_names()


    def check_version(self, version):
        if version.lower() not in self.data_versions:
            errmsg = f"Invalid data version: `{version}`. Available versions are: {self.data_versions}"
            raise ValueError(errmsg)


    def _version_is_same_or_higher(self, version):
        vmajor, vminor = [int(n) for n in self.version.strip('v').split('.')]
        other_vmajor, other_vminor = [int(n) for n in version.strip('v').split('.')]
        same_or_higher_version = False
        if other_vmajor >= vmajor:
            if other_vminor >= vminor:
                same_or_higher_version = True
        return same_or_higher_version


    def get_compatible_version(self, available_versions, description,
                               allow_higher_version=True):
        compatible_version = None
        if self.version in available_versions:
            compatible_version = self.version
        elif allow_higher_version:
            for v in available_versions:
                if self._version_is_same_or_higher(v):
                    compatible_version = v
        if compatible_version is None:
            if allow_higher_version:
                vinfo = f"version `'{available_versions[0]}'` or higher"
            else:
                versions = ', '.join([f"`'{v}'`" for v in available_versions])
                vinfo = f"one of the following versions: {versions}"
            errmsg = (f"No {description} available for version "
                      f"`'{self.version}'`. You must use {vinfo}.")
            raise NotImplementedError(errmsg)
        return compatible_version


    def get_freq(self, freq):
        if freq not in self.freqs:
            if '90' in str(freq):
                freq = 'f090'
            elif ('150' in str(freq)) or ('148' in str(freq)):
                freq = 'f150'
            else:
                raise ValueError(f"Invalid frequency: `freq = {freq}`. Options are: {self.freqs}")
        return freq

    
    # binning:

    def bin_edges_fname(self):
        """Returns the absolute path to the file holding the bin edges."""
        version = self.get_compatible_version(self.binning_versions, 'binning file')
        return self.binning_path(f'bin_edges_{version}.txt')
   

    def bin_edges(self):
        """Returns an array of bin edges."""
        bin_edges = np.loadtxt(self.bin_edges_fname())
        return bin_edges

    
    def binning_matrix(self, lmin=None, lmax=None):
        """Create a (num_bins, num_ells) binning matrix, which will bin
        the values between `lmin` and `lmax` in a vector/matrix
        containing values for each multipole between 2 and `lmax` values.
        For example, for an array `c_ell` holding a power spectrum with a
        value at each multipole `ell` in the range [2, 5000], to bin only
        the values in the range [30, 3000], you would pass `lmin = 30` and
        `lmax = 3000`.

        Note that many of the methods defined here return arrays starting
        from a multipole `ell=0`; to apply the binning matrix to these 
        arrays, you must start the arrays at the `ell=2` element.

        Parameters
        ----------
        lmin, lmax : int or None, default=None
            The minimum and maximum multipole values of the quantity to
            be binned, i.e. only values between `lmin` and `lmax` will be
            binned. If `lmin` or `lmax` is `None`, the default values for
            CMB-HD are used. 

        Returns
        -------
        binmat : array of float
            The two-dimensional binning matrix of shape (num_bins, num_ells).

        Examples
        --------
        >>> import numpy as np
        >>> from hd_mock_data import hd_data
        >>> lmax = 5000
        >>> ells = np.arange(lmax+1)
        >>> binmat = hd_data.HDMockData().binning_matrix(lmax=lmax)
        >>> binned_ells = binmat @ ells[2:]
        """
        if lmin is None:
            lmin = self.lmin
        if lmax is None:
            lmax = self.lmax
        bin_edges = self.bin_edges()
        bin_lmax = int(bin_edges[-1])
        if lmax > bin_lmax:
            errmsg = (f"The requested `lmax = {lmax}` is too high for version "
                      f"`'{self.version}'`; the binning is stored up to "
                      f"`lmax = {bin_lmax}`.")
            raise ValueError(errmsg)
        bmat = binning_matrix(bin_edges, lmin=lmin, lmax=lmax, start_at_ell=2)
        return bmat


    # theory spectra
    
    def cmb_theory_fname(self, cmb_type, baryonic_feedback=False,
                         pol_only_lensing=False, use_class=False):
        """Returns the name of the file containing the theory CMB and lensing
        spectra.

        Parameters
        ----------
        cmb_type : str
            The name of the kind of CMB spectra. Must be either `'lensed'`,
            `'delensed'`, or `'unlensed'`.
        baryonic_feedback : bool, default=False
            If `True`, the file name returned will be for a file holding
            theory calculated with the HMCode2020 + baryonic feedback
            non-linear model, as opposed to the HMCode2016 CDM-only model.
        pol_only_lensing : bool, default=False
            If `True`, the lensing reconstruction noise used to delens
            the CMB power spectra was calculated with only the EE and EB
            estimators. By default, the TT, TE, TB, EE, and EB estimators
            are used. Only an option for delensed power spectra and data
            versions >= 1.2.
        use_class : bool, default=False
            If `True`, the theory was calculated using CLASS; by default,
            it is calculated using CAMB. Only an option for lensed or
            unlensed power spectra and data versions >= 1.2.

        Returns
        -------
        fname : str
            The absolute path and name of the requested file.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed, or if
            `cmb_type='delensed'` and `use_class=True`.

        Note
        ----
        The file will have a column for the multipoles of the spectra, the
        CMB TT, TE, EE, and BB power spectra (in units of uK^2, without
        any multiplicative factors applied), and the lensing power spectrum,
        using the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where
        L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.
        """
        cmb_type = cmb_type.lower()
        if cmb_type not in self.cmb_types:
            raise ValueError(f"Unknown `cmb_type`: `'{cmb_type}'`. The `cmb_type` "
                             f"must be one of: {self.cmb_types}.")
        version = self.get_compatible_version(self.theo_versions, f'{cmb_type} theory spectra')
        if use_class:
            version = self.get_compatible_version(self.class_theo_versions, 'CLASS theory power spectra')
            if cmb_type == 'delensed':
                raise ValueError(f"`{cmb_type = }` and `{use_class = }`. CLASS does not calculate "
                                 "delensed power spectra; you must pass `cmb_type='lensed'` or "
                                 "`cmb_type='unlensed'` to load in CLASS theory power spectra.")
            theo_info = f'{cmb_type}_CLASS'
        elif (cmb_type == 'delensed') and pol_only_lensing:
            version = self.get_compatible_version(self.nlkk_pol_versions, 'polarization-only lensing')
            theo_info = f'{cmb_type}_MVpol'
        else:
            theo_info = cmb_type
        lmin = self.lmin
        lmax = self.theo_lmax
        fname = f'hd_lmin{lmin}lmax{lmax}_{theo_info}_cls_{version}.txt'
        if baryonic_feedback:
            return self.cdm_baryons_theo_path(fname)
        else:
            return self.cdm_theo_path(fname)


    def cmb_theory_spectra(self, cmb_type, baryonic_feedback=False, output_lmax=None,
                           pol_only_lensing=False, use_class=False):
        """Returns a dictionary containing the theory CMB and lensing spectra.

        Parameters
        ----------
        cmb_type : str
            The name of the kind of CMB spectra. Must be either `'lensed'`,
            `'delensed'`, or `'unlensed'`.
        baryonic_feedback : bool, default=False
            If `True`, the file name returned will be for a file holding
            theory calculated with the HMCode2020 + baryonic feedback
            non-linear model, as opposed to the HMCode2016 CDM-only model.
        pol_only_lensing : bool, default=False
            If `True`, the lensing reconstruction noise used to delens
            the CMB power spectra was calculated with only the EE and EB
            estimators. By default, the TT, TE, TB, EE, and EB estimators
            are used. Only an option for `cmb_type='delensed'` and data
            versions >= 1.2.
        use_class : bool, default=False
            If `True`, the theory was calculated using CLASS; by default,
            it is calculated using CAMB. Only an option for lensed or
            unlensed power spectra and data versions >= 1.2.

        Returns
        -------
        theo : dict of array of float
            A dictionary with a key `'ells'` holding the multipoles for the
            power spectra; keys `'tt'`, `'te'`, `'ee'`, and `'bb'` for the
            CMB power spectra for the requested `cmb_type`; and a key`'kk'`
            for the CMB lensing spectrum.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed, or if
            `cmb_type='delensed'` and `use_class=True`.

        Note
        ----
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum uses the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4,
        where L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.
        """
        fname = self.cmb_theory_fname(cmb_type, baryonic_feedback=baryonic_feedback,
                                      pol_only_lensing=pol_only_lensing,
                                      use_class=use_class)
        theo = load_from_file(fname, self.theo_cols)
        if output_lmax is not None:
            theo_lmax = int(theo['ells'][-1])
            output_lmax = int(output_lmax)
            if output_lmax > theo_lmax:
                msg = (f"The requested `output_lmax = {output_lmax}` is "
                       "higher than the maximum multipole of the spectra. "
                       f"Returning spectra up to `lmax = {theo_lmax}`.")
                warnings.warn(msg)
        else:
            output_lmax = self.lmaxTT
        for key in theo.keys():
            theo[key] = theo[key][:output_lmax+1]
        return theo
    
            
    def mcmc_bandpowers_fname(self, cmb_type, baryonic_feedback=False,
                              pol_only_lensing=False, use_class=False):
        """Returns the absolute path to the file containing the MCMC bandpowers.

        Parameters
        ----------
        cmb_type : str
            The name of the kind of CMB spectra. Must be either `'lensed'`,
            `'delensed'`, or `'unlensed'`.
        baryonic_feedback : bool, default=False
            If `True`, the file name returned will be for a file holding
            theory calculated with the HMCode2020 + baryonic feedback
            non-linear model, as opposed to the HMCode2016 CDM-only model.
        pol_only_lensing : bool, default=False
            If `True`, the lensing reconstruction noise used to delens
            the CMB power spectra was calculated with only the EE and EB
            estimators. By default, the TT, TE, TB, EE, and EB estimators
            are used. Only an option for `cmb_type='delensed'` and data
            versions >= 1.2.
        use_class : bool, default=False
            If `True`, the theory was calculated using CLASS; by default,
            it is calculated using CAMB. Only an option for lensed or
            unlensed power spectra and data versions >= 1.2.

        Returns
        -------
        fname : str
            The absolute path and name of the requested file.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed, or if
            `cmb_type='delensed'` and `use_class=True`.

        Note
        ----
        The bandpowers are the binned theory spectra stored as a single column,
        in the order TT, TE, EE, BB, kappakappa.
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum uses the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4,
        where L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.
        """
        cmb_type = cmb_type.lower()
        if cmb_type not in self.cmb_types[:-1]:
            errmsg = (f"Invalid `cmb_type`: `'{cmb_type}'`. The `cmb_type` "
                      f"must be one of: {self.cmb_types[:-1]}.")
            raise ValueError(errmsg)
        version = self.get_compatible_version(self.mcmc_bandpower_versions, f'{cmb_type} MCMC bandpowers')
        if use_class:
            version = self.get_compatible_version(self.class_theo_versions, 'CLASS theory power spectra')
            if cmb_type == 'delensed':
                raise ValueError(f"`{cmb_type = }` and `{use_class = }`. CLASS does not calculate "
                                 "delensed power spectra; you must pass `cmb_type='lensed'` to "
                                 "load in CLASS bandpowers.")
            theo_info = f'{cmb_type}_CLASS'
        elif (cmb_type == 'delensed') and pol_only_lensing:
            version = self.get_compatible_version(self.nlkk_pol_versions, 'polarization-only lensing')
            theo_info = f'{cmb_type}_MVpol'
        else:
            theo_info = cmb_type
        fname = f'hd_lmin{self.lmin}lmax{self.lmax}_{theo_info}_bandpowers_mcmc_{version}.txt'
        if baryonic_feedback:
            return self.cdm_baryons_theo_path(fname)
        else:
            return self.cdm_theo_path(fname)


    def mcmc_bandpowers(self, cmb_type, baryonic_feedback=False,
                        pol_only_lensing=False, use_class=False):
        """Returns an array holding the MCMC bandpowers.

        Parameters
        ----------
        cmb_type : str
            The name of the kind of CMB spectra. Must be either `'lensed'`,
            `'delensed'`, or `'unlensed'`.
        baryonic_feedback : bool, default=False
            If `True`, returns bandpowers calculated with the HMCode2020
            + baryonic feedback non-linear model, as opposed to the
            HMCode2016 CDM-only model.
        pol_only_lensing : bool, default=False
            If `True`, the lensing reconstruction noise used to delens
            the CMB power spectra was calculated with only the EE and EB
            estimators. By default, the TT, TE, TB, EE, and EB estimators
            are used. Only an option for `cmb_type='delensed'` and data
            versions >= 1.2.
        use_class : bool, default=False
            If `True`, the theory was calculated using CLASS; by default,
            it is calculated using CAMB. Only an option for lensed or
            unlensed power spectra and data versions >= 1.2.

        Returns
        -------
        bandpowers : array of float
            The binned CMB and lensing theory, stored as a single 1D array,
            with the binned theory stacked in the order TT, TE, EE, BB, kk.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed, or if
            `cmb_type='delensed'` and `use_class=True`.

        Note
        ----
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum uses the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4,
        where L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.
        """
        fname = self.mcmc_bandpowers_fname(cmb_type, baryonic_feedback=baryonic_feedback,
                                           pol_only_lensing=pol_only_lensing, use_class=use_class)
        bandpowers = np.loadtxt(fname)
        return bandpowers

    
    # FG:
    
    def fg_spectra_fname(self, freq):
        """Returns the name of the file containing the residual
        extragalactic foreground power spectra at the given frequency 
        for CMB-HD.

        Parameters
        ----------
        frequency : str or int
            Pass `90` or `'f090'` for a file containing columns for the
            different foreground components at 90 GHz, or pass `150` or
            `'f150'` for the corresponding file at 150 GHz.

        Returns
        -------
        fname : str
            The file name (including its absolute path).

        Raises
        ------
        ValueError
            If an invalid `frequency` was passed.

        Notes
        -----
        For version 1.2, the residual extragalactic foreground power
        spectra is obtained from simulations by applying the
        foreground-cleaning procedure described in arXiv:2609.16128.
        Since CIB and radio sources are removed simultaneously, we
        provide the sum of the residual CIB and radio sources, instead
        of separate CIB and radio power spectra. Since the kSZ signal is
        not removed, we do not include the kSZ power spectrum in the file.
        """
        freq = self.get_freq(freq)
        version = self.get_compatible_version(self.fg_versions, 'foreground spectra')
        fname = f'cmbhd_fg_cls_{freq}_{version}.txt'
        return self.fg_path(fname)


    def fg_spectra(self, freq, output_lmax=None):
        """Returns a dictionary holding power spectra of residual extragalactic
        foregrounds in temperature at the given frequency for CMB-HD.

        Parameters
        ----------
        frequency : str or int
            The frequency for the foreground power spectra. Pass `90` or
            `'f090'` for the foreground components at 90 GHz, or pass `150`
            or  `'f150'` for the foregrounds at 150 GHz.
        output_lmax : int or None, default=None
            If provided, cut the spectra at a maximum multipole given by the
            `output_lmax` value.

        Returns
        -------
        fgs : dict of array_like of float
            A dictionary of one-dimensional arrays with the following
            keys and values:
            - `'ells'` : The multipoles of the power spectra, starting
                         at zero.
            - `'ksz'` : The kSZ power spectrum (reionzation-only for
                        version 1.0, total reionization + late-time
                        otherwise).
            - `'tsz'` : The residual tSZ power spectrum.
            * In versions 1.0 and 1.1:
              - `'cib'` : The residual CIB power spectrum.
              - `'radio'` : The residual radio power spectrum.
            * In version 1.2:
              - `'cib_radio'` : The total residual CIB and radio power
                                spectrum.

        Raises
        ------
        ValueError
            If an invalid `frequency` was passed.

        Notes
        -----
        The power spectra are in units of uK^2, without any multiplicative
        factors applied.

        For version 1.2, the residual extragalactic foreground power
        spectra is obtained from simulations by applying the
        foreground-cleaning procedure described in arXiv:2609.16128.
        Since CIB and radio sources are removed simultaneously, we
        provide the sum of the residual CIB and radio sources, instead
        of separate CIB and radio power spectra. The simlation-based
        power spectra are originally binned, so the returned power
        spectra have been interpolated to each multipole.
        """
        fname = self.fg_spectra_fname(freq)
        fg = load_from_file(fname, self.fg_cols)
        fg_lmax = int(fg['ells'][-1])
        if 'ksz' not in fg:
            _, fg['ksz'] = self.cl_ksz(output_lmax=fg_lmax)
        if output_lmax is not None:
            output_lmax = int(output_lmax)
            if output_lmax > fg_lmax:
                msg = (f"The requested `output_lmax = {output_lmax}` is "
                       "higher than the maximum multipole of the spectra. "
                       f"Returning spectra up to `lmax = {fg_lmax}`.")
                warnings.warn(msg)
        else:
            output_lmax = self.lmaxTT
        for key in fg.keys():
            fg[key] = fg[key][:output_lmax+1]
        return fg


    def coadded_fg_spectrum_fname(self):
        """Returns the name of the file containing the coadded foreground
        power spectrum for the combination of 90 and 150 GHz for CMB-HD.

        Returns
        -------
        fname : str
            The file name (including its absolute path).
        """
        version = self.get_compatible_version(self.fg_versions, 'coadded foreground spectrum')
        fname = f'cmbhd_coadd_f090f150_total_fg_cls_{version}.txt'
        return self.fg_path(fname)


    def coadded_fg_spectrum(self, output_lmax=None):
        """Returns the power spectrum of the residual extragalactic foregrounds
        in temperature for CMB-HD, coadded from 90 and 150 GHz.

        Parameters
        ----------
        output_lmax : int or None, default=None
            If provided, cut the spectra at a maximum multipole given by the
            `output_lmax` value.

        Returns
        -------
        ells, coadd_fg_cls : array_like of float
            One-dimensional arrays holding the multipoles of the foreground
            power spectrum (`ells`) and the coadded foreground power spectrum
            (`coadd_fg_cls`).

        Note
        ----
        The power spectrum is in units of uK^2, without any multiplicative
        factors applied.
        """
        fname = self.coadded_fg_spectrum_fname()
        ells, cl_fg = np.loadtxt(fname, unpack=True)
        if output_lmax is not None:
            fg_lmax = int(ells[-1])
            output_lmax = int(output_lmax)
            if output_lmax > fg_lmax:
                msg = (f"The requested `output_lmax = {output_lmax}` is "
                       "higher than the maximum multipole of the coadded "
                       "foreground spectrum. Returning the spectrum up to "
                       f"`lmax = {fg_lmax}`.")
                warnings.warn(msg)
        else:
            output_lmax = self.lmaxTT
        ells = ells[:output_lmax+1]
        cl_fg = cl_fg[:output_lmax+1]
        return ells, cl_fg
        
        
    def cl_ksz_fname(self):
        """Returns the name of the file holding the kSZ power spectrum."""
        version = self.get_compatible_version(self.cl_ksz_versions, 'total kSZ power spectrum')
        fname = f'cmbhd_total_ksz_cls_{version}.txt'
        return self.fg_path(fname)

    
    def cl_ksz(self, output_lmax=None):
        """Returns a tuple of arrays holding the kSZ power spectrum and 
        the corresponding multipoles.

        Parameters
        ----------
        output_lmax : int or None, default=None
            If provided, cut the spectra at a maximum multipole given by the
            `output_lmax` value.

        Returns
        -------
        ells, cl_ksz : array_like of float
            One-dimensional arrays holding the multipoles of the kSZ power
            spectrum (`ells`) and the kSZ power spectrum itself (`cl_ksz`).

        Note
        ----
        The power spectrum is in units of uK^2, without any multiplicative
        factors applied.
        """
        fname = self.cl_ksz_fname()
        ells, cl_ksz = np.loadtxt(fname, unpack=True)
        if output_lmax is not None:
            ksz_lmax = int(ells[-1])
            output_lmax = int(output_lmax)
            if output_lmax > ksz_lmax:
                msg = (f"The requested `output_lmax = {output_lmax}` is "
                       "higher than the maximum multipole of the kSZ "
                       "power spectrum. Returning the spectrum up to "
                       f"`lmax = {ksz_lmax}`.")
                warnings.warn(msg)
        else:
            output_lmax = self.lmaxTT
        ells = ells[:output_lmax+1]
        cl_ksz = cl_ksz[:output_lmax+1]
        return ells, cl_ksz


    # noise spectra:

    def white_noise_cls(self, freq, output_lmax=None):
        """Returns a dictionary of the beam-deconvolved instrumental noise 
        spectra for CMB-HD TT, TE, EE, and BB power spectra at the given 
        frequency.
        
        Parameters
        ----------
        frequency : str or int
            The frequency for the noise power spectra. Pass `90` or `'f090'` 
            for the instrumental noise at 90 GHz, or pass `150` or `'f150'` 
            for the instrumental noise at 150 GHz.
        output_lmax : int or None, default=None
            If provided, cut the spectra at a maximum multipole given by the
            `output_lmax` value.

        Returns
        -------
        noise : dict of array_like of float
            A dictionary with a key `'ells'` whose value is a one-dimensional
            array holding the multipoles for the noise spectra, and keys `'tt'`,
            `'te'`, `'ee'`, and `'bb'` for one-dimensional arrays holding the
            corresponding noise power spectra.

        Note
        ----
        The noise spectra are in units of uK^2, without any multiplicative
        factors applied.
        """
        freq = self.get_freq(freq)
        if output_lmax is None:
            ells = self.ells.copy()
        else:
            ells = np.arange(output_lmax+1)
        # beam:
        theta_fwhm = np.deg2rad(self.beam_fwhm[freq] / 60) # arcmin -> radian
        beam = np.exp(-1. * (theta_fwhm * ells)**2 / (16. * np.log(2.)))
        # white noise:
        noise_level_temp = np.deg2rad(self.noise_levels[freq] / 60) # arcmin -> rad
        noise_level_pol = np.sqrt(2.) * noise_level_temp
        nls = {}
        nls['tt'] = (noise_level_temp / beam)**2
        nls['te'] = np.zeros(nls['tt'].shape)
        nls['ee'] = (noise_level_pol / beam)**2
        nls['bb'] = (noise_level_pol / beam)**2
        # for BB, we need to use the ASO noise below ell = 1000:
        theta_fwhm_aso = np.deg2rad(self.aso_beam_fwhm[freq] / 60)
        beam_aso = np.exp(-1. * (theta_fwhm_aso * ells)**2 / (16. * np.log(2.)))
        noise_level_pol_aso = np.sqrt(2) * np.deg2rad(self.aso_noise_levels[freq] / 60)
        nlbb_aso = (noise_level_pol_aso / beam_aso)**2
        nls['bb'][:1000] = nlbb_aso[:1000].copy()
        for key in nls.keys():
            nls[key][:2] = 0
        nls['ells'] = ells
        return nls


    def noise_cls_fname(self, freq):
        """Returns the name of the file containing the total noise on the
        CMB power spectra for CMB-HD at a given frequency, including
        residual extragalactic foregrounds in temperature.

        Parameters
        ----------
        frequency : str or int
            Pass `90` or `'f090'` for a file containing columns for the
            different foreground components at 90 GHz, or pass `150` or
            `'f150'` for the corresponding file at 150 GHz.

        Returns
        -------
        fname : str
            The file name (including its absolute path).

        Raises
        ------
        ValueError
            If an invalid `frequency` was passed.
        """
        freq = self.get_freq(freq)
        version = self.get_compatible_version(self.fg_versions, 'CMB noise + foreground spectra')
        fname = f'cmbhd_total_noise_fg_cls_{freq}_{version}.txt'
        return self.noise_path(fname)


    def noise_cls(self, freq, output_lmax=None, include_fg=True):
        """Returns a dictionary holding the noise on the CMB power spectra
        for CMB-HD at the given frequency, with or without residual
        extragalactic foregrounds in temperature.

        Parameters
        ----------
        frequency : str or int
            The frequency for the power spectra. Pass `90` or `'f090'`
            for the noise at 90 GHz, or pass `150` or `'f150'` for the
            noise at 150 GHz.
        output_lmax : int or None, default=None
            If provided, cut the spectra at a maximum multipole given by the
            `output_lmax` value.
        include_fg : bool, default=True
            If `True`, the temperature noise power spectrum includes
            the instrumental noise and the residual extragalactic foregrounds.
            If `False`, it will only contain instrumental noise.

        Returns
        -------
        noise : dict of array of float
            A dictionary with a key `'ells'` whose value is a one-dimensional
            array holding the multipoles for the noise spectra, and keys `'tt'`,
            `'te'`, `'ee'`, and `'bb'` for one-dimensional arrays holding the
            corresponding noise power spectra.

        Raises
        ------
        ValueError
            If an invalid `frequency` was passed.

        See Also
        --------
        cmb_noise_spectra : Coadded 90+150 GHz CMB noise power spectra.

        Notes
        -----
        The power spectra are in units of uK^2, without any multiplicative
        factors applied.

        In version 1.2, the temperature power spectrum is obtained by
        applying the foreground-cleaning procedure described in
        arXiv:2609.16128 to maps with the lensed CMB, white noise, kSZ,
        tSZ, CIB, and radio galaxies, and then taking the power spectrum
        of the foreground-cleaned temperature map after subtracting the
        lensed CMB realization from the map. Note that this will not
        precisely equal the sum of the residual foreground spectra and
        temperature white noise spectrum returned by `fg_spectra` and
        `white_noise_cls`, respectively, due to correlations between the
        different components in the map. The simlation-based temperature
        power spectrum is originally binned, so it has been interpolated
        to each multipole. The polarization power spectra are calculated
        for the CMB-HD noise levels using the `white_noise_cls` method.
        """
        if not include_fg:
            noise = self.white_noise_cls(freq, output_lmax=output_lmax)
        else:
            fname = self.noise_cls_fname(freq)
            noise = load_from_file(fname, self.noise_cols)
            noise_lmax = int(noise['ells'][-1])
            output_lmax = int(output_lmax) if (output_lmax is not None) else self.lmaxTT
            if output_lmax > noise_lmax:
                msg = (f"The requested `output_lmax = {output_lmax}` is higher than the maximum "
                       f"multipole of the spectra. Returning spectra up to `lmax = {noise_lmax}`.")
                warnings.warn(msg)
            for key in noise.keys():
                noise[key] = noise[key][:output_lmax+1]
        return noise

    
    def cmb_noise_fname(self, include_fg=True):
        """Returns the name of the file containing the power spectra of 
        the noise on the CMB TT, TE, EE, and BB spectra, coadded from 90 
        and 150 GHz.
        
        Parameters
        ----------
        include_fg : bool, default=True
            If `True`, the temperature noise in the returned file is the sum of
            the instrumental noise and the residual extragalactic foreground
            power spectrum. If `False`, it will only contain instrumental noise.

        Returns
        -------
        fname : str
            The name of the file holding the requested noise spectra.

        Note
        ----
        The returned file will have a column for the multipoles of the spectra,
        and columns for the CMB TT, TE, EE, and BB noise spectra (in units
        of uK^2, without any multiplicative factors applied).
        """
        if include_fg:
            version = self.get_compatible_version(self.cmb_noise_versions, f'coadded CMB noise + foregrounds')
            lmax = self.noise_lmax
            fg_info = 'withfg'
        else:
            version = self.get_compatible_version(self.cmb_white_noise_versions, f'coadded CMB white noise')
            lmax = self.cmb_white_noise_lmax
            fg_info = 'nofg'
        fname = f'hd_coaddf090f150_cmb_noise_cls_lmax{lmax}_{fg_info}_{version}.txt'
        return self.noise_path(fname)
        

    def cmb_noise_spectra(self, include_fg=True, output_lmax=None):
        """Returns a dictionary containing the power spectra of the noise 
        on the CMB TT, TE, EE, and BB spectra, coadded from 90 and 
        150 GHz, and the corresponding multipoles.
        
        Parameters
        ----------
        include_fg : bool, default=True
            If `True`, the temperature noise power spectrum includes
            the instrumental noise and the residual extragalactic foregrounds.
            If `False`, it will only contain instrumental noise.
        output_lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by the
            `output_lmax` value.

        Returns
        -------
        noise : dict of array of float
            A dictionary with a key `'ells'` whose value is a one-dimensional
            array holding the multipoles for the noise spectra, and keys `'tt'`,
            `'te'`, `'ee'`, and `'bb'` for one-dimensional arrays holding the
            corresponding noise power spectra.

        See Also
        --------
        noise_cls : 
            The CMB noise, with or without residual extragalactic 
            foregrounds, at 90 or 150 GHz.

        Note
        ----
        The noise spectra are in units of uK^2, without any multiplicative
        factors applied.
        """
        fname = self.cmb_noise_fname(include_fg=include_fg)
        noise = load_from_file(fname, self.noise_cols)
        if output_lmax is not None:
            noise_lmax = int(noise['ells'][-1])
            output_lmax = int(output_lmax)
            if output_lmax > noise_lmax:
                msg = (f"The requested `output_lmax = {output_lmax}` is "
                       "higher than the maximum multipole of the spectra. "
                       f"Returning spectra up to `lmax = {noise_lmax}`.")
                warnings.warn(msg)
        else:
            output_lmax = self.lmaxTT
        for key in noise.keys():
            noise[key] = noise[key][:output_lmax+1]
        return noise


    def lensing_noise_fname(self, pol_only_lensing=False):
        """The CMB lensing noise file name.

        Parameters
        ----------
        pol_only_lensing : bool, default=False
            If `True`, the lensing reconstruction noise was calculated
            with only the EE and EB estimators. By default, the TT, TE,
            TB, EE, and EB estimators are used. Only an option for data
            versions >= 1.2.

        Returns
        -------
        str
            The absolute path to the file holding the CMB lensing noise.
        """
        version = self.get_compatible_version(self.lensing_noise_versions, 'lensing noise')
        if pol_only_lensing:
            version = self.get_compatible_version(self.nlkk_pol_versions, 'polarization-only lensing')
        nlkk_info = 'nlkk_MVpol' if pol_only_lensing else 'nlkk'
        ell_info = f'lmin{self.nlkk_lmin}lmax{self.nlkk_lmax}Lmax{self.nlkk_Lmax}'
        fname = f'hd_{ell_info}_{nlkk_info}_{version}.txt'
        return self.noise_path(fname)


    def lensing_noise_spectrum(self, output_Lmax=None, pol_only_lensing=False):
        """Returns the CMB lensing noise spectrum and the corresponding
        lensing multipoles.

        Parameters
        ----------
        output_Lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by the
            `output_Lmax` value.
        pol_only_lensing : bool, default=False
            If `True`, the lensing reconstruction noise was calculated
            with only the EE and EB estimators. By default, the TT, TE,
            TB, EE, and EB estimators are used. Only an option for data
            versions >= 1.2.

        Returns
        -------
        L, nlkk : array_like of float
            One-dimensional arrays containing the lensing multipoles (`L`)
            and the lensing noise spectrum (`nlkk`).

        Note
        ----
        The CMB lensing noise N_L^kk is the noise on the CMB lensing spectrum
        C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where C_L^phiphi is the CMB
        lensing potential power spectrum and L is the lensing multipole.
        """
        fname = self.lensing_noise_fname(pol_only_lensing=pol_only_lensing)
        L, nlkk = np.loadtxt(fname, unpack=True)
        if output_Lmax is not None:
            noise_Lmax = int(L[-1])
            output_Lmax = int(output_Lmax)
            if output_Lmax > noise_Lmax:
                msg = (f"The requested `output_Lmax = {output_Lmax}` is "
                       "higher than the maximum multipole of the lensing "
                       f"noise. Returning noise up to `Lmax = {noise_Lmax}`.")
                warnings.warn(msg)
        else:
            output_Lmax = self.Lmax
        L = L[:output_Lmax+1]
        nlkk = nlkk[:output_Lmax+1]
        return L, nlkk


    # covmats:

    def block_covmat_fname(self, cmb_type, pol_only_lensing=False):
        """Returns the name of the file holding the covariance matrix for the
        mock CMB-HD TT, TE, EE, BB and CMB lensing power spectra for the
        the given CMB type (lensed or delensed).

        Parameters
        ----------
        cmb_type : str
            If `cmb_type='delensed'`, the file holds a covariance matrix for
            delensed CMB TT, TE, EE, and BB power spectra, in addition to the
            CMB lensing spectrum. If `cmb_type='lensed'`, the covariance matrix
            is for lensed CMB spectra instead, but otherwise includes the same
            set of power spectra as the delensed case.
        pol_only_lensing : bool, default=False
            If `True`, the covariance matrix was calculated with only the
            EE and EB estimators used for the lensing reconstruction
            noise (which will also change the delensed power spectra).
            By default, the TT, TE, TB, EE, and EB estimators are used.
            Only an option for data versions >= 1.2.

        Returns
        -------
        fname : str
            The name of the file that contains the requested covariance matrix.
        """
        if cmb_type.lower() not in self.cmb_types[:-1]:
            errmsg = (f"Invalid `cmb_type`: `'{cmb_type}'`. The `cmb_type` "
                     f"must be one of: {self.cmb_types[:-1]}.")
            raise ValueError(errmsg)
        version = self.get_compatible_version(self.covmat_versions, f'{cmb_type} full covariance matrix')
        if pol_only_lensing:
            version = self.get_compatible_version(self.nlkk_pol_versions, 'polarization-only lensing')
        lmin = self.covmat_lmin
        lmax = self.covmat_lmax
        cmb_type = cmb_type.lower()
        fsky = 'pt'.join(str(round(self.fsky,3)).split('.'))
        cov_info = 'MVpol_cov' if pol_only_lensing else 'cov'
        fname = f'hd_fsky{fsky}_lmin{lmin}lmax{lmax}_binned_{cmb_type}_{cov_info}_{version}.txt'
        return self.covmat_path(fname)


    def block_covmat(self, cmb_type, pol_only_lensing=False):
        """Returns the covariance matrix for the mock lensed or delensed
        CMB TT, TE, EE, BB and CMB lensing power spectra.

        Parameters
        ----------
        cmb_type : str
            If `cmb_type='delensed'`, returns a covariance matrix for delensed
            CMB TT, TE, EE, and BB power spectra, in addition to the CMB
            lensing spectrum. If `cmb_type='lensed'`, the covariance matrix is
            for lensed CMB spectra instead, but otherwise includes the same
            set of power spectra as the delensed case.
        pol_only_lensing : bool, default=False
            If `True`, the covariance matrix was calculated with only the
            EE and EB estimators used for the lensing reconstruction
            noise (which will also change the delensed power spectra).
            By default, the TT, TE, TB, EE, and EB estimators are used.
            Only an option for data versions >= 1.2.

        Returns
        -------
        covmat : array of float
            A two-dimensional array holding the full covariance matrix for the
            mock CMB power spectra.

        Note
        ----
        The covariance matrix is binned and contains 25 blocks; each block
        has shape `(nbin, nbin)`, where `nbin` is the number of bins in the
        multipole range for CMB-HD. The diagonal blocks contain
        the covariance matrices for TT x TT, TE x TE, EE x EE, BB x BB, and
        kk x kk, where kk refers to the CMB lensing spectrum. The off-diagonal
        blocks contain the cross-covariances, e.g. TT x TE, TT x EE, etc.
        We use units of  uK^2 for the CMB spectra, and do not apply any
        multiplicative factors. For the CMB lensing spectrum, we use the
        convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where C_L^phiphi is
        the CMB lensing potential power spectrum and L is the lensing multipole.
        """
        fname = self.block_covmat_fname(cmb_type, pol_only_lensing=pol_only_lensing)
        covmat = np.loadtxt(fname)
        return covmat


    def tt_diag_covmat_fname(self, cmb_type):
        """Returns the name of the file holding the diagonal covariance matrix 
        for the mock CMB-HD lensed or delensed TT power spectrum in the 
        multipole range from 20,100 to 40,000.

        Parameters
        ----------
        cmb_type : str
            Either `'lensed'` or `'delensed'`.

        Returns
        -------
        fname : str
            The name of the file that contains the requested covariance matrix.
        """
        if cmb_type.lower() not in self.cmb_types[:-1]:
            errmsg = (f"Invalid `cmb_type`: `'{cmb_type}'`. The `cmb_type` "
                     f"must be one of: {self.cmb_types[:-1]}.")
            raise ValueError(errmsg)
        version = self.get_compatible_version(self.tt_covmat_versions, 
                                              f'{cmb_type} TT x TT diagonal covariance matrix',
                                              allow_higher_version=False)
        lmin = self.tt_covmat_lmin
        lmax = self.tt_covmat_lmax
        cmb_type = cmb_type.lower()
        fsky = 'pt'.join(str(round(self.fsky,3)).split('.'))
        fname = f'hd_fsky{fsky}_lmin{lmin}lmax{lmax}_binned_{cmb_type}_TTxTT_cov_{version}.txt'
        return self.covmat_path(fname)


    def tt_diag_covmat(self, cmb_type):
        """Returns the  diagonal covariance matrix for the mock CMB-HD lensed 
        or delensed TT power spectrum in the multipole range from 20,100 to 40,000.

        Parameters
        ----------
        cmb_type : str
            Either `'lensed'` or `'delensed'`.

        Returns
        -------
        covmat : array of float
            A two-dimensional array holding the diagonal covariance matrix for the
            TT power spectrum.
        """
        fname = self.tt_diag_covmat_fname(cmb_type)
        covmat = np.loadtxt(fname)
        return covmat
    

    # theory code settings:

    def camb_settings_fname(self, baryonic_feedback=False, use_H0=False,
                            v2_camb_names=None):
        """Path to the file that contains CAMB parameters (cosmology,
        accuracy, etc.).

        Parameters
        ----------
        baryonic_feedback : bool, default=False
            If `True`, the file name returned will be for a file holding
            settings for the HMCode2020 + baryonic feedback non-linear
            model, as opposed to the HMCode2016 CDM-only model.
        use_H0 : bool, default=False
            Whether to use the Hubble constant `H0` instead of the
            cosmoMC theta parameter `cosmomc_theta`.

        Returns
        -------
        str
            The path to the file.

        Other Parameters
        ----------------
        v2_camb_names : bool or None, default=None
            If you are using CAMB version 2.0.0 or higher, pass
            `v2_camb_names=True`; if you are using a lower version, pass
            `v2_camb_names=False`. By default, when `v2_camb_names=None`,
            we attempt to import CAMB; if CAMB is installed,
            `v2_camb_names` is set based on the CAMB version. Otherwise,
            it will be set to `True`. The correct CAMB version is
            required to ensure that the correct parameter names are
            passed to CAMB.

        See Also
        --------
        camb_settings : Dictionary of CAMB parameters.
        """
        version = self.get_compatible_version(self.camb_theo_versions, 'CAMB parameters')
        H0info = '_useH0' if use_H0 else ''
        # some CAMB names were changed in version 2.0.0:
        if v2_camb_names is None:
            if self._use_v2camb is None:
                # warn the user about CAMB version:
                _camb_param_names_warning('parameter file')
                # set the `_use_v2camb` attribute,
                # so this warning is only raised once:
                self._use_v2camb = True
            v2_camb_names = self._use_v2camb
        camb_name = 'camb2' if v2_camb_names else 'camb'
        fname = f'{camb_name}_params{H0info}_{version}.yaml'
        if baryonic_feedback:
            return self.cdm_baryons_theo_path(fname)
        else:
            return self.cdm_theo_path(fname)


    def camb_settings(self, baryonic_feedback=False, use_H0=False,
                      v2_camb_names=None):
        """Dictionary of CAMB parameters (cosmology, accuracy, etc.).

        Parameters
        ----------
        baryonic_feedback : bool, default=False
            If `True`, the file name returned will be for a file holding
            settings for the HMCode2020 + baryonic feedback non-linear
            model, as opposed to the HMCode2016 CDM-only model.
        use_H0 : bool, default=False
            Whether to use the Hubble constant `H0` instead of the
            cosmoMC theta parameter `cosmomc_theta`.

        Returns
        -------
        params : dict
            A dictionary of CAMB settings.

        Other Parameters
        ----------------
        v2_camb_names : bool or None, default=None
            If you are using CAMB version 2.0.0 or higher, pass
            `v2_camb_names=True`; if you are using a lower version, pass
            `v2_camb_names=False`. By default, when `v2_camb_names=None`,
            we attempt to import CAMB; if CAMB is installed,
            `v2_camb_names` is set based on the CAMB version. Otherwise,
            it will be set to `True`. The correct CAMB version is
            required to ensure that the correct parameter names are
            passed to CAMB.

        Notes
        -----
        The returned `params` dict can be passed to the `camb.set_params`
        function, e.g. `pars = camb.set_params(**params)`.
        """
        # raise the warning about the CAMB version here, if necessary:
        if v2_camb_names is None:
            if self._use_v2camb is None:
                # warn the user about CAMB version:
                _camb_param_names_warning('dictionary')
                # set the `_use_v2camb` attribute,
                # so this warning is only raised once:
                self._use_v2camb = True
            v2_camb_names = self._use_v2camb
        fname = self.camb_settings_fname(baryonic_feedback=baryonic_feedback,
                                         use_H0=use_H0, v2_camb_names=v2_camb_names)
        with open(fname, 'r') as f:
            params = yaml.safe_load(f)
        return params


    def class_settings_fname(self, baryonic_feedback=False, use_H0=False):
        """Path to the file that contains CLASS parameters (cosmology,
        accuracy, etc.).

        Parameters
        ----------
        baryonic_feedback : bool, default=False
            If `True`, the file name returned will be for a file holding
            settings for the HMCode2020 + baryonic feedback non-linear
            model, as opposed to the HMCode2016 CDM-only model.
        use_H0 : bool, default=False
            Whether to use the Hubble constant `H0` instead of 
            `theta_s_100`.

        Returns
        -------
        str
            The path to the file.

        See Also
        --------
        class_settings : Dictionary of CLASS parameters.

        Notes
        -----
        The file does not contain the path to the `sBBN file` provided 
        with `hdMockData` (because the absolute path cannot be determined
        prior to installing this code). Use the `class_settings` method
        to load the YAML file in to a dictionary, and add the correct path
        to the `sBBN file`.
        """
        version = self.get_compatible_version(self.class_theo_versions, 'CLASS parameters')
        H0info = '_useH0' if use_H0 else ''
        fname = f'class_params{H0info}_{version}.yaml'
        if baryonic_feedback:
            return self.cdm_baryons_theo_path(fname)
        else:
            return self.cdm_theo_path(fname)


    def class_settings(self, baryonic_feedback=False, use_H0=False):
        """Dictionary of CLASS parameters (cosmology, accuracy, etc.).

        Parameters
        ----------
        baryonic_feedback : bool, default=False
            If `True`, the file name returned will be for a file holding
            settings for the HMCode2020 + baryonic feedback non-linear
            model, as opposed to the HMCode2016 CDM-only model.
        use_H0 : bool, default=False
            Whether to use the Hubble constant `H0` instead of 
            `theta_s_100`.

        Returns
        -------
        params : dict
            A dictionary of CLASS settings.

        Notes
        -----
        The CLASS settings include `accurate_lensing=1`; when this is
        used, CLASS cannot calculate the power spectra past a maximum
        multipole of about 14,000, which is lower than the value of
        `l_max_scalars` used (given by the `theo_lmax` attribute).
        See arXiv:YYYY.YYYYY (!! TODO:LINK2ZACK !! ) for instructions to
        modify CLASS so that a higher `l_max_scalars` can be used with
        `accurate_lensing=1`.

        The returned `params` dict can be passed to the `classy.Class.set`
        method, e.g. by calling `cosmo = classy.Class()` and
        `cosmo.set(params)`.
        """
        fname = self.class_settings_fname(baryonic_feedback=baryonic_feedback, 
                                          use_H0=use_H0)
        with open(fname, 'r') as f:
            params = yaml.safe_load(f)
        params['sBBN file'] = self.class_sbbn_file
        params['l_max_scalars'] = self.theo_lmax + 500
        # warn about the need to modify class:
        msg = ("By default, CLASS cannot calculate the power spectra with "
               f"`accurate_lensing = {params['accurate_lensing']}` and "
               f"`l_max_scalars` = {self.theo_lmax+500}.") # TODO : add ref. to paper for instructions
        warnings.warn(msg)
        return params

