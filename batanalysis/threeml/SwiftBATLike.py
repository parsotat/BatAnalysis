from pathlib import Path
from typing import Optional, Union
import collections


import numpy as np
import astropy.io.fits as fits

from astromodels import Parameter, Model

from threeML.plugin_prototype import PluginPrototype
from threeML.utils.OGIP.response import OGIPResponse
from threeML.plugins.SpectrumLike import SpectrumLike
from threeML.plugins.XYLike import XYLike
from threeML.utils.spectrum.pha_spectrum import PHASpectrum
from threeML.utils.OGIP.pha import PHAII, PHAWrite
from threeML.plugins.OGIPLike import OGIPLike


import logging
logger = logging.getLogger(__name__)

#try to import BatAnalysis objects for object comparison later on
from batanalysis.bat_tte import BatEvent
from batanalysis.batproducts import Spectrum as ba_spectrum
from batanalysis.bat_drm import BatDRM

__instrument_name = "Swift BAT"

class SwiftBATLike(OGIPLike):
    """
    Plugin for Swift BAT data analysis in the 3ML framework
    """
    def __init__(
        self,
        name: str,
        observation: Union[str, Path, PHASpectrum, PHAII, BatEvent, ba_spectrum],
        background: Optional[
            Union[str, Path, PHASpectrum, PHAII, SpectrumLike, XYLike]
        ] = None,
        response: Optional[str] = None,
        nuisance_params: Optional[Union[Parameter, list[Parameter]]]=None,
        spectrum_number: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize the Swift BAT plugin.

        Parameters:
        -----------
        name : str
            Name for this instance
        observation : str or numpy.ndarray
            Either the BAT data file path or a pre-loaded spectrum array
        background : str or numpy.ndarray, optional
            Background spectrum file or array
        response : str, optional
            Path to the response (RSP) file
        nuisance_params : astromodels.core.parameter.Parameter or list[astromodels.core.parameter.Parameter], optional
            list of nuisance parameters relevant for the Swift BAT instrument.
            Examples could include:
            - 'bkg_norm': Background normalization factor
            - 'sys_error': Systematic error percentage
            - 'gain_shift': Gain shift correction factor
        is_weighted : bool, optional
            If True, use Gaussian likelihood (weighted least squares) for fitting.
            If False, use Poisson likelihood. Default is True.
        """

        # create the hash for the nuisance parameters. We have none for now.
        self._nuisance_parameters = collections.OrderedDict()

        # Initialize internal variables
        self._fit_nuisance_params = False
        self._verbose = verbose

        # Set the likelihood type based on is_weighted
        #TODO: have self._is_weighted be set when the pha file is passed to _load_bat_data and determine it from the file
        self._is_weighted = False
        if self._is_weighted:
            self._likelihood_type = "Gaussian"
        else:
            self._likelihood_type = "Poisson"

        # Process nuisance parameters
        self._setup_nuisance_parameters(nuisance_params)

        # Load the observation/response/background data
        self._load_bat_data(name, observation=observation, response=response, background=background)

        #setup a dictionary to hold source model values
        self._source_location = {}


    def _setup_nuisance_parameters(self, nuisance_params):
        """
        Set up nuisance parameters for the Swift BAT instrument

        Parameters
        ----------
        nuisance_params : dict or None
            Dictionary with nuisance parameters and their initial values/bounds
        """
        # Default nuisance parameters if None provided
        if nuisance_params is None:
            self.set_inner_minimization(False)
        elif isinstance(nuisance_param, Parameter):
            self.set_inner_minimization(True)
            self._nuisance_parameters[self.nuisance_param.name] = self.nuisance_param
            self._nuisance_parameters[self.nuisance_param.name].free = self._fit_nuisance_params
        elif isinstance(nuisance_param, list):
            self.set_inner_minimization(True)
            test = [isinstance(i, Parameter) for i in nuisance_param]
            if np.any(test):
                raise RuntimeError(
                    "Nuisance parameter must be astromodels.core.parameter.Parameter object or a list of astromodels.core.parameter.Parameter objects")

            for i in nuisance_param:
                self._nuisance_parameters[i.name] = i
                self._nuisance_parameters[i.name].free = self._fit_nuisance_params
        else:
            raise RuntimeError(
                "Nuisance parameter must be astromodels.core.parameter.Parameter object or a list of astromodels.core.parameter.Parameter objects")

    def set_inner_minimization(self, flag):
        """
        Turn on the minimization of the internal BAT (nuisance) parameters.

        Parameters
        ----------
        flag : bool
            Turns on and off the minimization  of the internal parameters
        """

        self._fit_nuisance_params: bool = bool(flag)

        for parameter in self._nuisance_parameters:
            self._nuisance_parameters[parameter].free = self._fit_nuisance_params

    def _load_bat_data(self,
           name: str,
           observation: Union[str, Path, PHASpectrum, BatEvent, ba_spectrum],
           background: Optional[
               Union[str, Path, PHASpectrum, SpectrumLike, XYLike]
           ] = None,
           response: Optional[str] = None,
           nuisance_params: Optional[Union[Parameter, list[Parameter]]] = None,
           ):

        if isinstance(observation, BatEvent):
            raise NotImplementedError("The use of BatEvent data with the Swift-BAT plugin is not yet implemented.")

        if background is not None:
            raise NotImplementedError("The inclusion of backgrounds with Swift BAT data, assuming that the data is not "
                                      "mask-weighted, is not yet implemented.")

        #if we have a BatSpectrum object, its easiest to extract the path to the pha file. We should also do this for
        # the response file since the OGIPLike class will attempt to read the response file from the PHA file and will
        # not be able to actually find it

        # TODO: in the future if one wants to do non-mask weighted analyses then a response can be generated on the fly,
        #   especially if the astromodels model is being fitted for a position. this is only relevant for TTE data

        if isinstance(observation, ba_spectrum):
            observation=ba_spectrum.pha_file

            if response is None:
                response=ba_spectrum.drm_file

        #if we have a string or path for the PHA file and no response file is specified
        # then assume that the response file is the same base name as the pha file with ".rsp" instead of ".pha"
        if (isinstance(observation, Path) or isinstance(observation, str)) and response is None:
            if isinstance(observation, str):
                observation=Path(observation)

            # in the future can do below, but right now the Spectrum object is only compatible with TTE data and not
            # survey data, so fo rnow just construct the response name manually
            #spectrum=ba_spectrum.from_file(pha_file=observation)
            #response=ba_spectrum.drm_file
            response = observation.parent.joinpath(f"{observation.stem}.rsp")

        if isinstance(observation, PHASpectrum) and response is None:
            response=observation.response_file

        if response is None:
            raise ValueError("A response has not been specified for the SwiftBATLike plugin to use.")

        # cant have mask weighted pha file and a background file too
        if self._is_weighted and background is not None:
            raise RuntimeError("It is not possible to have a mask weighted observation and include background.")

        # Initialize base class
        super(SwiftBATLike, self).__init__(name, observation=observation, background=background, response=response,
                                           verbose=self._verbose)

    def set_model(self, likelihoodmodel: Model):

        #for now we call the OGIPLike set_model method, which by default doesnt handle extended sources.
        #TODO: in the future can attempt to incorporate extended source models to this

        # Get point sources and extended sources from model:
        point_sources = likelihoodmodel.point_sources
        extended_sources = likelihoodmodel.extended_sources

        # Source counter for models with multiple sources:
        self.src_counter = 0

        #check that we only have a single point source model since we are only dealing with the mask weighted spectrum,
        # also catches the case of no point sources specified.
        if len(point_sources) != 1:
            raise NotImplementedError("Dealing with non-mask weighted spectra with more than one astromodels model is "
                                      "not yet implemented.")

        # Initialization
        # should only be applicable if we are starting off new or changing the number of point sources we have.
        # Currently, we only accept a single point source for the mask weighted spectrum
        if len(point_sources) != 0:

            if len(self._source_location) == 0 or len(point_sources) != len(self._source_location):

                for name, source in point_sources.items():
                    coord = source.position.sky_coord

                    self._source_location[name] = coord.copy()  # to avoid same memory issue

                    logger.info(f"saved source name : {name})")

                logger.info(f"Done saving the model information")

        #check to make sure that the location of the source hasnt changed since we arent supporting that yet
        # TODO: to support this, we need to be able to recalculate the response on the fly
        if np.any([source.position.sky_coord != self._source_location[name] for name, source in point_sources.items()]):
            raise NotImplementedError("It is not possible to change the position of the source model in the likelihood calculation.")

        super(SwiftBATLike,self).set_model(likelihoodmodel)

    def get_log_like(self):
        # for now just call the parent class method
        #TODO: can have changes for different statistics for mask weighted versus non-mask weighted spectra
        return super(SwiftBATLike,self).get_log_like()

    def inner_fit(self) -> float:
        #TODO: add capability to integrate likelihood over nusiance parameters
        if self._fit_nuisance_params:
            raise NotImplementedError("Tt is not possible to consider the nuisance paramters in the calculation of the loglikelihood.")

        return self.get_log_like()


class SwiftBATLike2(PluginPrototype):
    """
    Plugin for Swift BAT data analysis in the 3ML framework
    """
    
    def __init__(
        self,
        name: str,
        observation: Union[str, Path, PHASpectrum, PHAII, BatEvent, ba_spectrum],
        background: Optional[
            Union[str, Path, PHASpectrum, PHAII, SpectrumLike, XYLike]
        ] = None,
        response: Optional[Union[str, BatDRM]] = None,
        nuisance_params: Optional[Union[Parameter, list[Parameter]]]=None,
        is_weighted=True,
    ):
        """
        Initialize the Swift BAT plugin.
        
        Parameters:
        -----------
        name : str
            Name for this instance
        observation : str or numpy.ndarray
            Either the BAT data file path or a pre-loaded spectrum array
        background : str or numpy.ndarray, optional
            Background spectrum file or array
        response : str, optional
            Path to the response (RSP) file
        nuisance_params : astromodels.core.parameter.Parameter or list[astromodels.core.parameter.Parameter], optional
            list of nuisance parameters relevant for the Swift BAT instrument.  
            Examples could include:
            - 'bkg_norm': Background normalization factor
            - 'sys_error': Systematic error percentage
            - 'gain_shift': Gain shift correction factor
        is_weighted : bool, optional
            If True, use Gaussian likelihood (weighted least squares) for fitting.
            If False, use Poisson likelihood. Default is True.
        """
        
        # create the hash for the nuisance parameters. We have none for now.
        self._nuisance_parameters = collections.OrderedDict()
        
        # Initialize base class
        super(SwiftBATLike, self).__init__(name, self._nuisance_parameters)
        
        # Initialize internal variables
        self._fit_nuisance_params = False
        self._observation_spectrum = None
        self._background_spectrum = None
        self._response_matrix = None
        self._ebounds = None
        self._observed_counts = None
        self._background_counts = None
        self._verbose = True
        
        # Set the likelihood type based on is_weighted
        self._is_weighted = is_weighted
        if self._is_weighted:
            self._likelihood_type = "Gaussian"
            self._observation_errors = None  # Will be loaded with data or calculated
        else:
            self._likelihood_type = "Poisson"
        
        # Process nuisance parameters
        self._setup_nuisance_parameters(nuisance_params)
        
        # Load the observation data
        self._load_bat_data(observation)

        # Load the background if provided
        if self._is_weighted and background is not None:
            raise RuntimeError("It is not possible to have a mask weighted observation and include background.")

        self._load_bat_background(background)

        # Load the response matrix otherwise throw an error.
        # TODO: put the error in the _load_response_matrix method
        # TODO: in the future if one wants to do non-mask weighted analyses then a response can be generated on the fly,
        #   especially if the astromodels model is being fitted for a position. this is only relevant for TTE data
        #if response is not None:
        self._load_response_matrix(response)
        #else:
        #    raise ValueError("Response file must be provided for Swift BAT analysis")
            
        # Set energy boundaries
        self._energies = np.array(self._response_matrix.ebounds)
        
        # Set default energy range for BAT (typically 15-150 keV)
        self._active_measurements = np.ones(self._energies.shape[0], dtype=bool)
        
        # Set initial mask for the energy range
        self.set_active_measurements('15-150')
        
        # Calculate errors for weighted likelihood if needed
        if self._is_weighted and self._observation_errors is None:
            self._calculate_errors()
        
    def _setup_nuisance_parameters(self, nuisance_params):
        """
        Set up nuisance parameters for the Swift BAT instrument

        Parameters
        ----------
        nuisance_params : dict or None
            Dictionary with nuisance parameters and their initial values/bounds
        """
        # Default nuisance parameters if None provided
        if nuisance_params is None:
            self.set_inner_minimization(False)
        elif isinstance(nuisance_param, Parameter):
            self.set_inner_minimization(True)
            self._nuisance_parameters[self.nuisance_param.name] = self.nuisance_param
            self._nuisance_parameters[self.nuisance_param.name].free = self._fit_nuisance_params
        elif isinstance(nuisance_param, list):
            self.set_inner_minimization(True)
            test=[isinstance(i, Parameter) for i in nuisance_param]
            if np.any(test):
                raise RuntimeError("Nuisance parameter must be astromodels.core.parameter.Parameter object or a list of astromodels.core.parameter.Parameter objects")
            
            for i in nuisance_param:
                self._nuisance_parameters[i.name] = i
                self._nuisance_parameters[i.name].free = self._fit_nuisance_params
        else:
            raise RuntimeError("Nuisance parameter must be astromodels.core.parameter.Parameter object or a list of astromodels.core.parameter.Parameter objects")
        
    def _load_bat_data(self, data_file):
        """Load Swift BAT data from a FITS file"""
        with fits.open(data_file) as hdul:
            # Assuming standard BAT PHA format - adjust according to your specific files
            self._observed_counts = hdul['SPECTRUM'].data['COUNTS'].astype(float)
            
            # Try to load statistical errors for weighted likelihood
            if self._is_weighted:
                try:
                    self._observation_errors = hdul['SPECTRUM'].data['STAT_ERR'].astype(float)
                except (KeyError, AttributeError):
                    # Will calculate errors later if not found
                    pass
            
            # Extract energy boundaries
            if 'EBOUNDS' in hdul:
                e_min = hdul['EBOUNDS'].data['E_MIN']
                e_max = hdul['EBOUNDS'].data['E_MAX']
                self._ebounds = (e_min, e_max)
            else:
                # Default energy bounds might need to be set based on your BAT configuration
                raise ValueError("Could not find energy bounds in the BAT data file")
                
    def _load_bat_background(self, background_file):
        """Load Swift BAT background from a FITS file"""
        with fits.open(background_file) as hdul:
            self._background_counts = hdul['SPECTRUM'].data['COUNTS'].astype(float)
    
    def _calculate_errors(self):
        """
        Calculate statistical errors for the observed counts.
        For Gaussian likelihood (weighted), we need error estimates.
        """
        # For Poisson data, the error is sqrt(N)
        # For low count rates, can use different approaches
        self._observation_errors = np.sqrt(np.maximum(self._observed_counts, 1.0))
        
        if self._verbose:
            print("Calculated statistical errors for weighted likelihood")
            
    # Required methods from PluginPrototype
    def set_model(self, model):
        """
        Set the model to be used for this dataset
        
        Parameters:
        -----------
        model: astromodels.Model
            Model instance
        """
        self._model = model
        
        # Figure out which source in the model is to be used
        if self._source_name is not None:
            # Use the provided source
            try:
                self._source = self._model[self._source_name]
            except KeyError:
                raise KeyError(f"Source {self._source_name} is not in the model")
        else:
            # Only one source in the model
            if len(list(self._model.point_sources.keys())) == 1:
                self._source = list(self._model.point_sources.values())[0]
            else:
                raise RuntimeError("This plugin needs a source name to be specified")
                
    def get_log_like(self):
        """
        Return the logarithm of the likelihood for the current set of parameters
        """
        # Get the expected counts from the model
        expected_counts = self._get_expected_model_counts()
        
        # Apply nuisance parameters
        # 1. Gain shift (energy scale adjustment)
        if 'gain_shift' in self._nuisance_parameters:
            gain = self._parameters['gain_shift'].value
            # Apply gain shift (in a real implementation, this would require more sophisticated handling)
            # This is a simplified example
            if gain != 1.0:
                # Placeholder for gain shift implementation
                pass
        
        # 2. Background normalization
        if self._background_counts is not None and 'bkg_norm' in self._parameters:
            bkg_norm = self._parameters['bkg_norm'].value
            expected_counts_with_bkg = expected_counts + (self._background_counts * bkg_norm)
        else:
            expected_counts_with_bkg = expected_counts
        
        # Apply mask for active measurements
        active_obs = self._observed_counts[self._active_measurements]
        active_exp = expected_counts_with_bkg[self._active_measurements]
        
        # 3. Apply systematic error (if specified)
        sys_err_fraction = 0
        if 'sys_error' in self._parameters:
            sys_err_percentage = self._parameters['sys_error'].value
            sys_err_fraction = sys_err_percentage / 100.0
        
        # Choose likelihood based on is_weighted parameter
        if self._is_weighted:
            # Gaussian likelihood (weighted least squares)
            
            # Get the errors for the active measurements
            active_err = self._observation_errors[self._active_measurements]
            
            # Add systematic error contribution in quadrature
            total_err = np.sqrt(active_err**2 + (sys_err_fraction * active_exp)**2)
            
            # Calculate Gaussian log-likelihood
            # ln(L) = -0.5 * sum((observed - expected)^2 / sigma^2) - sum(ln(sigma))
            chi2 = np.sum(((active_obs - active_exp) / total_err)**2)
            norm_term = np.sum(np.log(2 * np.pi * total_err**2))
            
            log_like = -0.5 * (chi2 + norm_term)
            
        else:
            # Poisson likelihood
            if sys_err_fraction > 0:
                # For Poisson with systematics, use a modified likelihood
                sys_err_term = np.sum(
                    np.power((active_obs - active_exp) / (active_exp * sys_err_fraction), 2)
                )
                log_like = np.sum(active_obs * np.log(active_exp) - active_exp - self._log_factorial(active_obs))
                log_like -= 0.5 * sys_err_term
            else:
                # Standard Poisson likelihood
                log_like = np.sum(active_obs * np.log(active_exp) - active_exp - self._log_factorial(active_obs))
        
        return log_like
    
    def _log_factorial(self, n):
        """Calculate log(n!) using Stirling's approximation for large values"""
        n = np.array(n, dtype=float)
        idx = (n > 0)
        
        res = np.zeros_like(n)
        # Use stirling approximation for values > 100 to avoid numerical issues
        mask = (n[idx] > 100)
        
        if np.any(mask):
            m = n[idx][mask]
            res[idx][mask] = m * np.log(m) - m + 0.5 * np.log(2 * np.pi * m)
        
        # Use gamma for smaller values
        if np.any(~mask & idx):
            from scipy.special import gammaln
            m = n[idx][~mask]
            res[idx][~mask] = gammaln(m + 1)
        
        return res
    
    def _get_expected_model_counts(self):
        """
        Get the expected counts from the model folded through the response
        """
        # Get the differential flux from the model (ph/cm^2/s/keV)
        differential_flux = self._get_diff_flux_from_model()
        
        # Multiply by the response to get expected counts
        return self._response_matrix.fold_spectrum(differential_flux)
        
    def _get_diff_flux_from_model(self):
        """
        Get the differential flux predicted by the model at the response energies
        """
        energies = self._response_matrix.monte_carlo_energies
        
        # Use the point source spectrum to get the differential flux
        return self._source.get_spectral_model()(energies)
    
    def set_active_measurements(self, e_min=None, e_max=None):
        """
        Set which energy channel/measurements to include in the analysis
        
        Parameters:
        -----------
        e_min: float or str, optional
            The minimum energy to include
        e_max: float or str, optional
            The maximum energy to include
        """
        # If string is passed like "15-150", parse it
        if isinstance(e_min, str) and e_max is None:
            tokens = e_min.split("-")
            e_min, e_max = float(tokens[0]), float(tokens[1])
            
        # Get channel-energy mapping
        ebounds = self._response_matrix.ebounds
        
        # Create the mask
        if e_min is not None:
            idx_min = np.searchsorted(ebounds[:, 0], e_min)
        else:
            idx_min = 0
            
        if e_max is not None:
            idx_max = np.searchsorted(ebounds[:, 1], e_max)
        else:
            idx_max = len(ebounds)
            
        # Set the active measurements mask
        self._active_measurements = np.zeros(self._observed_counts.shape, dtype=bool)
        self._active_measurements[idx_min:idx_max] = True
        
        if self._verbose:
            print(f"Active energy range set to {ebounds[idx_min, 0]}-{ebounds[idx_max-1, 1]} keV")
    
    def inner_fit(self):
        """
        Inner fit method called by the fit engine
        Returns:
            float: negative log likelihood value
        """
        return self.get_log_like() * (-1)
    
    def get_likelihood_type(self):
        """
        Return the likelihood type (Gaussian or Poisson)
        """
        return self._likelihood_type
        
    # Additional methods specific to Swift BAT
    def display_bat_diagnostics(self):
        """Display BAT-specific diagnostic plots"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # Example: plot raw counts
        plt.step(np.arange(len(self._observed_counts)),
                 self._observed_counts,
                 where='mid', label='BAT Observation')
        
        if self._background_counts is not None:
            plt.step(np.arange(len(self._background_counts)),
                     self._background_counts,
                     where='mid', label='Background', linestyle='--')
            
        # If using weighted likelihood, show error bars
        if self._is_weighted and hasattr(self, '_observation_errors'):
            plt.errorbar(np.arange(len(self._observed_counts)),
                         self._observed_counts,
                         yerr=self._observation_errors,
                         fmt='none', ecolor='k', alpha=0.3)
            
        plt.xlabel('Channel')
        plt.ylabel('Counts')
        plt.legend()
        plt.title(f'Swift BAT Raw Spectrum - {self.name} ({self._likelihood_type} likelihood)')
        plt.tight_layout()
        plt.show()
    
    def view_count_spectrum(self):
        """Display the count spectrum with background if available"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Get the mid-point of energy bins
        e_mid = np.array([(e[0] + e[1])/2.0 for e in self._response_matrix.ebounds])
        
        # Apply masks for active energy range
        e_mid_active = e_mid[self._active_measurements]
        counts_active = self._observed_counts[self._active_measurements]
        
        # Plot the data
        if self._is_weighted and hasattr(self, '_observation_errors'):
            # With error bars
            errors_active = self._observation_errors[self._active_measurements]
            ax.errorbar(e_mid_active, counts_active, yerr=errors_active,
                      fmt='o', label='BAT Data')
        else:
            # Without error bars
            ax.step(e_mid_active, counts_active, where='mid', label='BAT Data')
        
        # Plot background if available
        if self._background_counts is not None:
            bkg_active = self._background_counts[self._active_measurements]
            ax.step(e_mid_active, bkg_active, where='mid', linestyle='--', label='Background')
            
            # Plot model background (if fitted)
            if 'bkg_norm' in self._parameters and hasattr(self, '_model'):
                bkg_norm = self._parameters['bkg_norm'].value
                ax.step(e_mid_active, bkg_active * bkg_norm, where='mid',
                        linestyle=':', label=f'Background Model (norm={bkg_norm:.2f})')
            
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Counts')
        ax.legend()
        ax.set_title(f'Swift BAT Spectrum - {self.name} ({self._likelihood_type} likelihood)')
        
        plt.tight_layout()
        plt.show()
        
    def set_likelihood_type(self, is_weighted):
        """
        Change the likelihood type after initialization
        
        Parameters:
        -----------
        is_weighted: bool
            If True, use Gaussian likelihood.
            If False, use Poisson likelihood.
        """
        self._is_weighted = is_weighted
        
        if self._is_weighted:
            self._likelihood_type = "Gaussian"
            # Calculate errors if not already available
            if not hasattr(self, '_observation_errors') or self._observation_errors is None:
                self._calculate_errors()
        else:
            self._likelihood_type = "Poisson"
            
        if self._verbose:
            print(f"Likelihood type set to {self._likelihood_type}")
            
    def set_inner_minimization(self, flag):
        """
        Turn on the minimization of the internal BAT (nuisance) parameters.
        
        Parameters
        ----------
        flag : bool
            Turns on and off the minimization  of the internal parameters
        """
        
        self._fit_nuisance_params: bool = bool(flag)

        for parameter in self._nuisance_parameters:
            self._nuisance_parameters[parameter].free = self._fit_nuisance_params
