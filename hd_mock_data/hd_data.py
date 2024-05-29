import os
import warnings
import numpy as np

def binning_matrix(bin_edges, lmin=None, lmax=None, start_at_ell=2):
    """Create a (num_bins, num_ells) binning matrix, which will bin the values
    between `lmin` and `lmax` in a vector/matrix containing values for each
    multipole between the `start_at_ell` and `lmax` values. For example, for
    an array `c_ell` holding a power spectrum with a value at each multipole
    `ell` in the range [2, 5000], to bin only the values in the range [30, 3000],
    you would pass `lmin = 30`, `lmax = 3000`, and `start_at_ell=2`.

    Parameters
    ----------
    bin_edges : array of int
        A one dimensional array holding the upper bin edge for each bin,
        except the first element, which is the lower bin edge of the first bin.
    lmin, lmax : int or None, default=None
        The minimum and maximum multipole values of the quantity to be binned,
        i.e. only values between `lmin` and `lmax` will be binned. If `lmin` is
        `None`, we use the first value in the `bin_edges` array; if `lmax`
        is `None`, we use the last value in the `bin_edges` array.
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


def get_version_number(version):
    return round(float(version[1:]), 2)


def get_compatible_version(version, available_versions):
    if version in available_versions:
        compatible_version = version
    else:
        version_num = get_version_number(version)
        compatible_version = None
        for v in available_versions:
            if version_num >= get_version_number(v):
                compatible_version = v
    return compatible_version


class HDMockData:
    def __init__(self, version='latest'):
        self.data_versions = ['v1.0', 'v1.1']
        self.latest_version = self.data_versions[-1]
        if 'late' in version.lower():
            self.version = self.latest_version
        else:
            self.check_version(version)
            self.version = version.lower()
        self.version_number = get_version_number(self.version)

        # keep track of versions for each kind of file:
        self.binning_versions = ['v1.0', 'v1.1']
        self.theo_versions = ['v1.0', 'v1.1']
        self.mcmc_bandpower_versions = ['v1.0', 'v1.1']
        self.fg_versions = ['v1.0', 'v1.1']
        self.cl_ksz_versions = ['v1.1']
        self.cmb_noise_versions = ['v1.0', 'v1.1'] # includes FG in TT
        self.cmb_white_noise_versions = ['v1.0'] # white noise only
        self.lensing_noise_versions = ['v1.0']
        self.covmat_versions = ['v1.0', 'v1.1'] # full 5 x 5 covmat, 30 < ell < 20k
        self.tt_covmat_versions = ['v1.1'] # diagonal TTxTT, 20k < ell < 40k

        # multipoles:
        self.lmin = 30
        self.lmax = 20100
        self.Lmin = 30
        self.Lmax = 20100
        if self.version_number > 1.0:
            self.lmaxTT = 40000
        else:
            self.lmaxTT = 20100
        # currently, white noise saved up to lmax = 40,000;
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
        
        self.fsky = 0.6
        self.ells = np.arange(self.lmaxTT + 1)
        self.theo_cols = ['ells', 'tt', 'te', 'ee', 'bb', 'kk']
        self.noise_cols = self.theo_cols[:-1]
        self.fg_cols = ['ells', 'ksz', 'tsz', 'cib', 'radio']
        self.cmb_types = ['lensed', 'delensed', 'unlensed']
        self.freqs = ['f090', 'f150']
        self.noise_levels = {'f090': 0.7, 'f150': 0.8} # uK-arcmin
        self.beam_fwhm = {'f090': 0.42, 'f150': 0.25} # arcmin
        self.aso_noise_levels = {'f090': 3.5, 'f150': 3.8} # uK-arcmin
        self.aso_beam_fwhm = {'f090': 2.2, 'f150': 1.4} # arcmin

        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
        self.data_path = lambda x: os.path.join(self.data_dir, x)
        self.binning_path = lambda x: os.path.join(self.data_path('binning'), x)
        self.theo_path = lambda x: os.path.join(self.data_path('theory'), x)
        self.cdm_theo_path = lambda x: os.path.join(self.theo_path('cdm'), x)
        self.cdm_baryons_theo_path = lambda x: os.path.join(self.theo_path('cdm_baryonic_feedback'), x)
        self.noise_path = lambda x: os.path.join(self.data_path('noise'), x)
        self.fg_path = lambda x: os.path.join(self.data_path('foregrounds'), x)
        self.covmat_path = lambda x: os.path.join(self.data_path('covariance_matrices'), x)


    def check_version(self, version):
        if version.lower() not in self.data_versions:
            errmsg = f"Invalid data version: `{version}`. Available versions are: {self.data_versions}"
            raise ValueError(errmsg)


    def set_version(self, version=None):
        if version is None:
            version = self.version
        else:
            self.check_version(version)
        return version


    def get_compatible_version(self, available_versions, description):
        if self.version in available_versions:
            compatible_version = self.version
        else:
            compatible_version = None
            for v in available_versions:
                if self.version_number >= get_version_number(v):
                    compatible_version = v
            if compatible_version is None:
                errmsg = (f"No {description} available for version "
                          f"`'{self.version}'`. You must use version "
                          f"`'{available_versions[0]}'` or higher.")
                raise NotImplementedError(errmsg)
        return compatible_version

    
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
        """Create a (num_bins, num_ells) binning matrix, which will bin the values
        between `lmin` and `lmax` in a vector/matrix containing values for each
        multipole between the `start_at_ell` and `lmax` values. For example, for
        an array `c_ell` holding a power spectrum with a value at each multipole
        `ell` in the range [2, 5000], to bin only the values in the range [30, 3000],
        you would pass `lmin = 30`, `lmax = 3000`, and `start_at_ell=2`.

        Parameters
        ----------
        lmin, lmax : int or None, default=None
            The minimum and maximum multipole values of the quantity to be binned,
            i.e. only values between `lmin` and `lmax` will be binned. If `lmin` 
            or `lmax` is `None`, the default values for CMB-HD are used. 
        start_at_ell : int, default=2
            The minimum multipole value in the quantity to be binned. This is
            typically either `0` or `2`.

        Returns
        -------
        binmat : array of float
            The two-dimensional binning matrix of shape (num_bins, num_ells).
        """
        if lmin is None:
            lmin = self.lmin
        if lmax is None:
            lmax = self.lmax
        if lmax > self.lmaxTT:
            errmsg = (f"The requested `lmax = {lmax}` is too high for version "
                      f"`'{self.version}'`; the binning is stored up to "
                      f"`lmax = {self.lmaxTT}`.")
            raise ValueError(errmsg)
        bin_edges = self.bin_edges()
        bmat = binning_matrix(bin_edges, lmin=lmin, lmax=lmax, start_at_ell=2)
        return bmat


    # theory spectra
    
    def cmb_theory_fname(self, cmb_type, baryonic_feedback=False):
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

        Returns
        -------
        fname : str
            The absolute path and name of the requested file.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed.

        Note
        ----
        The file will have a column for the multipoles of the spectra, the 
        CMB TT, TE, EE, and BB power spectra (in units of uK^2, without
        any multiplicative factors applied), and the lensing power spectrum,
        using the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where
        L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.
        """
        if cmb_type.lower() not in self.cmb_types:
            errmsg = (f"Unknown `cmb_type`: `'{cmb_type}'`. The `cmb_type` "
                     f"must be one of: {self.cmb_types}.")
            raise ValueError(errmsg)
        version = self.get_compatible_version(self.theo_versions, f'{cmb_type} theory spectra')
        lmin = self.lmin
        lmax = self.lmaxTT # theory saved up to max. value of lmax
        fname = f'hd_lmin{lmin}lmax{lmax}_{cmb_type.lower()}_cls_{version}.txt'
        if baryonic_feedback:
            return self.cdm_baryons_theo_path(fname)
        else:
            return self.cdm_theo_path(fname)


    def cmb_theory_spectra(self, cmb_type, baryonic_feedback=False, output_lmax=None):
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
            If an unrecognized `cmb_type` was passed.

        Note
        ----
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum uses the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4,
        where L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.
        """
        fname = self.cmb_theory_fname(cmb_type, baryonic_feedback=baryonic_feedback)
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
    
            
    def mcmc_bandpowers_fname(self, cmb_type, baryonic_feedback=False):
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

        Returns
        -------
        fname : str
            The absolute path and name of the requested file.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed.

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
        if cmb_type.lower() not in self.cmb_types[:-1]:
            errmsg = (f"Invalid `cmb_type`: `'{cmb_type}'`. The `cmb_type` "
                     f"must be one of: {self.cmb_types[:-1]}.")
            raise ValueError(errmsg)
        version = self.get_compatible_version(self.mcmc_bandpower_versions, f'{cmb_type} MCMC bandpowers')
        fname = f'hd_lmin{self.lmin}lmax{self.lmax}_{cmb_type.lower()}_bandpowers_mcmc_{version}.txt'
        if baryonic_feedback:
            return self.cdm_baryons_theo_path(fname)
        else:
            return self.cdm_theo_path(fname)

    
    def mcmc_bandpowers(self, cmb_type, baryonic_feedback=False):
        """Returns an array holding the MCMC bandpowers.
        
        Parameters
        ----------
        cmb_type : str
            The name of the kind of CMB spectra. Must be either `'lensed'`,
            `'delensed'`, or `'unlensed'`.
        baryonic_feedback : bool, default=False
            If `True`, the file name returned will be for a file holding 
            theory calculated with the HMCode2020 + baryonic feedback 
            non-linear model, as opposed to the HMCode2016 CDM-only model.

        Returns
        -------
        bandpowers : array of float
            The binned CMB and lensing theory, stored as a single 1D array,
            with the binned theory stacked in the order TT, TE, EE, BB, kk.

        Raises
        ------
        ValueError
            If an unrecognized `cmb_type` was passed.

        Note
        ----
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum uses the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4,
        where L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.
        """
        fname = self.mcmc_bandpowers_fname(cmb_type, baryonic_feedback=baryonic_feedback)
        bandpowers = np.loadtxt(fname)
        return bandpowers

    
    # FG:
    
    def fg_spectra_fname(self, freq):
        """Returns the name of the file containing the residual extragalactic
        foreground power spectra for CMB-HD.

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
        if freq not in self.freqs:
            if '90' in str(freq):
                freq = 'f090'
            elif '150' in str(freq):
                freq = 'f150'
            else:
                errmsg = (f"Invalid frequency: `freq = {freq}`. "
                          f"Options are: {self.freqs}")
                raise ValueError(errmsg)
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
            A dictionary of one-dimensional arrays with a key `'ells'` holding
            the multipoles of the power spectra, and keys `'ksz'`, `'tsz'`,
            `'cib'`, and `'radio'` holding the residual foreground power
            spectra for reionization kSZ, tSZ, CIB, and radio sources,
            respectively.

        Raises
        ------
        ValueError
            If an invalid `frequency` was passed.

        Note
        ----
        The power spectra are in units of uK^2, without any multiplicative
        factors applied.
        """
        fname = self.fg_spectra_fname(freq)
        fg = load_from_file(fname, self.fg_cols)
        if output_lmax is not None:
            fg_lmax = int(fg['ells'][-1])
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
        if freq not in self.freqs:
            if '90' in str(freq):
                freq = 'f090'
            elif '150' in str(freq):
                freq = 'f150'
            else:
                errmsg = (f"Invalid frequency: `freq = {freq}`. "
                          f"Options are: {self.freqs}")
                raise ValueError(errmsg)
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


    
    def cmb_noise_fname(self, include_fg=True):
        """Returns the name of the file containing the power spectra of the
        noise on the CMB TT, TE, EE, and BB spectra.
        
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
            lmax = self.lmaxTT
            fg_info = 'withfg'
        else:
            version = self.get_compatible_version(self.cmb_white_noise_versions, f'coadded CMB white noise')
            lmax = self.cmb_white_noise_lmax
            fg_info = 'nofg'
        fname = f'hd_coaddf090f150_cmb_noise_cls_lmax{lmax}_{fg_info}_{version}.txt'
        return self.noise_path(fname)
        

    def cmb_noise_spectra(self, include_fg=True, output_lmax=None):
        """Returns a dictionary containing the power spectra of the noise on 
        the CMB TT, TE, EE, and BB spectra, and the corresponding multipoles.
        
        Parameters
        ----------
        include_fg : bool, default=True
            If `True`, the temperature noise power spectrum is the sum of
            the instrumental noise and the residual extragalactic foreground
            power spectrum. If `False`, it will only contain instrumental noise.
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

        Note
        ----
        The noise spectra are in units of uK^2, without any multiplicative
        factors applied.
        """
        fname = self.cmb_noise_fname(include_fg=include_fg)
        noise = load_from_file(fname, self.theo_cols[:-1])
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


    def lensing_noise_fname(self):
        """Returns the absolute path to the file holding the CMB lensing noise."""
        version = self.get_compatible_version(self.lensing_noise_versions, 'lensing noise')
        fname = f'hd_lmin{self.nlkk_lmin}lmax{self.nlkk_lmax}Lmax{self.nlkk_Lmax}_nlkk_{version}.txt'
        return self.noise_path(fname)


    def lensing_noise_spectrum(self, output_Lmax=None):
        """Returns the CMB lensing noise spectrum and the corresponding
        lensing multipoles.

        Parameters
        ----------
        output_Lmax : int or None, default=None
            If provided, cut the spectrum at a maximum multipole given by the
            `output_Lmax` value.

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
        fname = self.lensing_noise_fname()
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

    def block_covmat_fname(self, cmb_type):
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
        lmin = self.covmat_lmin
        lmax = self.covmat_lmax
        cmb_type = cmb_type.lower()
        fname = f'hd_fsky0pt6_lmin{lmin}lmax{lmax}_binned_{cmb_type}_cov_{version}.txt'
        return self.covmat_path(fname)


    def block_covmat(self, cmb_type):
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
        fname = self.block_covmat_fname(cmb_type)
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
        version = self.get_compatible_version(self.tt_covmat_versions, f'{cmb_type} TT x TT diagonal covariance matrix')
        lmin = self.tt_covmat_lmin
        lmax = self.tt_covmat_lmax
        cmb_type = cmb_type.lower()
        fname = f'hd_fsky0pt6_lmin{lmin}lmax{lmax}_binned_{cmb_type}_TTxTT_cov_{version}.txt'
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
        
    

