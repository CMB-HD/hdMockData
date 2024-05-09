import os
import warnings
import numpy as np

def binning_matrix(bin_edges, lmin=None, lmax=None, start_at_ell=2):
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
        self.mcmc_bandpower_versions = ['v1.0']
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
        version = self.get_compatible_version(self.binning_versions, 'binning file')
        return self.binning_path(f'bin_edges_{version}.txt')
   

    def bin_edges(self):
        bin_edges = np.loadtxt(self.bin_edges_fname())
        return bin_edges

    
    def binning_matrix(self, lmin=None, lmax=None):
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
        fname = self.mcmc_bandpowers_fname(cmb_type, baryonic_feedback=baryonic_feedback)
        bandpowers = np.loadtxt(fname)
        return bandpowers

    
    # FG:
    
    def fg_spectra_fname(self, freq):
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
        version = self.get_compatible_version(self.fg_versions, 'coadded foreground spectrum')
        fname = f'cmbhd_coadd_f090f150_total_fg_cls_{version}.txt'
        return self.fg_path(fname)


    def coadded_fg_spectrum(self, output_lmax=None):
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
        version = self.get_compatible_version(self.cl_ksz_versions, 'total kSZ power spectrum')
        fname = f'cmbhd_total_ksz_cls_{version}.txt'
        return self.fg_path(fname)

    
    def cl_ksz(self, output_lmax=None):
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
        version = self.get_compatible_version(self.lensing_noise_versions, 'lensing noise')
        fname = f'hd_lmin{self.nlkk_lmin}lmax{self.nlkk_lmax}Lmax{self.nlkk_Lmax}_nlkk_{version}.txt'
        return self.noise_path(fname)


    def lensing_noise_spectrum(self, output_Lmax=None):
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
        nkkk = nlkk[:output_Lmax+1]
        return L, nlkk


    # covmats:

    def block_covmat_fname(self, cmb_type):
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
        fname = self.block_covmat_fname(cmb_type)
        return np.loadtxt(fname)


    def tt_diag_covmat_fname(self, cmb_type):
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
        fname = self.tt_diag_covmat_fname(cmb_type)
        return np.loadtxt(fname)
        
    

