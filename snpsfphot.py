from matplotlib import pyplot as plt, cm
import photutils
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from os import path
from numpy import meshgrid

from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.psf import IterativelySubtractedPSFPhotometry, BasicPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma

_HST_WFC3_PSF_FWHM_ARCSEC = 0.14  # FWHM of the HST WFC3IR PSF in arcsec


def centroid(imdat, x0, y0, boxsize=11):
    """ Locate a source in the image near position (x0,y0) using a 2-D gaussian 
    centroid algorithm. 
    :param imdat: image data (2-D array)
    :param x0: initial guess of x pixel location of the source 
    :param y0: initial guess of y pixel location of the source 
    :param boxsize: size of the box to search in (pixels) 
    :return: 
    """
    # create a mask that masks out all the pixels more than boxsize/2
    # pixels from the first-guess SN position (the center of the image,
    # or the target location specified by the user)
    halfbox = boxsize / 2.
    mask = np.ones(imdat.shape, dtype=bool)

    mask[int(y0 - halfbox):int(y0 + halfbox),
         int(x0 - halfbox):int(x0 + halfbox)] = False
    xnew, ynew = photutils.centroid_2dg(imdat, mask=mask)

    return xnew, ynew


class Photometry(object):

    def __init__(self, name):
        self.name = name
        self.psfphotobject = None
        self.psfmodel = None
        self.psfphotresults = None


class TargetImage(object):
    """An image containing an isolated point source to be photometered."""

    def __init__(self, imfilename):
        """ 
        :param imfilename: full path for a .fits image file
        """

        # read in the target image and determine the pixel scale (arcsec/pix)
        targetim = fits.open(imfilename)
        self.imdat = targetim[0].data
        self.wcs = wcs.WCS(targetim[0].header)
        self.pixscale = np.mean(
            wcs.utils.proj_plane_pixel_scales(self.wcs.celestial) * 3600)

        # initial guesses of position and flux of the target source
        self.x_0 = None
        self.y_0 = None
        self.flux_0 = None

        # Dictionary of Photometry objects. Each entry will hold a PSF model,
        # photometry object, and phot results table. Keyed with
        # user-specified names for the psf models.
        self.photometry = {}

    @property
    def target_table(self):
        """ Returns a Table object with the centroid coordinates and the required
        column names x_0 and y_0. We include a flux_0 column to
        give an initial guess for the flux of the star."""

        if self.flux_0 is None:
            target_table = Table(
                data=[[self.x_0], [self.y_0]], names=['x_0', 'y_0'])
        else:
            target_table = Table(
                data=[[self.x_0], [self.y_0], [self.flux_0]],
                names=['x_0','y_0','flux_0'])
        return target_table


    # TODO : allow multiple targets?
    def set_target(self, x_0=None, y_0=None, flux_0=None,
                   recenter=False):
        """set the pixel location and optionally the flux of the target"""
        # TODO: allow RA,Dec location
        if x_0 is None:
            self.x_0 = self.imdat.shape[0] / 2.
        else:
            self.x_0 = x_0
        if y_0 is None:
            self.y_0 = self.imdat.shape[1] / 2.
        else:
            self.y_0 = y_0

        if recenter:
            # Apply a centroiding algorithm to locate the source in the image
            self.x_0, self.y_0 = centroid(self.imdat, self.x_0, self.y_0)

        # If the user leaves flux0 as None then aperture photometry will
        # be used for the first guess at the flux of the target
        self.flux_0 = flux_0


    def load_psfmodel(self, psfimfilename, modelname=None, **kwargs):
        """Initialize an HSTPSFModel object to use for measuring photometry 
        :param psfimfilename: full path to the psf image fits file
        :param modelname: name of the model to store in the modeldict. If left 
         as None, the basename of the psf image file is used.
        Additional keyword args get passed to the HSTPSFModel initializer.
        :return:
        """
        if modelname is None:
            modelname = path.basename(psfimfilename)
        # TODO: special case for a Gaussian PRF

        # Load a psf model from a "psf star image"
        self.photometry[modelname] = Photometry(modelname)
        self.photometry[modelname].psfmodel = HSTPSFModel(
            psfimfilename, self.pixscale, **kwargs)


    def dophot(self, modelname, fitpix=11, apradpix=3,
               **kwargs):
        """Do photometry of the target: 
        set up a photometry object, do the photometry and store the results.
        """
        if modelname not in self.photometry:
            print("Model {} not loaded. Use load_psfmodel()".format(modelname))
            return
        hstpsfmodel = self.photometry[modelname].psfmodel

        # Make the photometry object
        hstphotobject = BasicPSFPhotometry(
            psf_model=hstpsfmodel.psfmodel, group_maker=hstpsfmodel.grouper,
            bkg_estimator=hstpsfmodel.bkg_estimator,
            fitter=hstpsfmodel.fitter, fitshape=(fitpix, fitpix),
            aperture_radius=apradpix)
        self.photometry[modelname].psfphotobject = hstphotobject

        phot_results_table = hstphotobject.do_photometry(
            image=self.imdat, init_guesses=self.target_table)
        self.photometry[modelname].psfphotresults = phot_results_table


    def plot_resid_image(self, modelname, Npix=15):
        """Make a figure showing the target, the psf model, and the residual 
        image after subtracting the PSF model. 
        Note: assumes that the photobject has already been used to collect
        photometry on a target image, so that the get_residual_image() method
        will return a valid resid image.
        TODO: make this more generalized!"""

        if modelname not in self.photometry:
            print("Model {} not loaded. Use load_psfmodel()"
                  " and then dophot() to run photometry.".format(modelname))
            return
        photinstance = self.photometry[modelname]
        if photinstance.psfphotresults is None:
            print("No phot results for model {}. "
                  " Use dophot()".format(modelname))
            return

        psfmodel = self.photometry[modelname].psfmodel
        results_table = self.photometry[modelname].psfphotresults
        targetimdat = self.imdat

        # TODO : read these from the results table?
        xtarget = self.x_0
        ytarget = self.y_0

        fig = plt.gcf()
        ax1 = fig.add_subplot(1, 3, 1)
        im1 = ax1.imshow(
            targetimdat[int(ytarget) - Npix / 2:int(ytarget) + Npix / 2,
            int(xtarget) - Npix / 2:int(xtarget) + Npix / 2],
            interpolation='nearest', cmap='viridis', origin='left')
        plt.colorbar(im1, orientation='horizontal', fraction=0.046, pad=0.1,
                     ticks=plt.MaxNLocator(5))

        # use np.meshgrid to define a coordinate grid over which the PSF model
        # can be evaluated.
        gridindices = np.meshgrid(np.arange(len(targetimdat)),
                                  np.arange(len(targetimdat)))
        parameters_to_set = {'x_fit': 'x_0', 'y_fit': 'y_0', 'flux_fit': 'flux'}

        photobject = self.photometry[modelname].psfphotobject
        groupmodel = photutils.psf.models.get_grouped_psf_model(
            photobject.psf_model, results_table, parameters_to_set)
        psf_image = groupmodel(gridindices[0], gridindices[1])

        ax2 = fig.add_subplot(1, 3, 2)
        im2 = ax2.imshow(psf_image[int(ytarget) - Npix / 2:int(ytarget) + Npix / 2,
                         int(xtarget) - Npix / 2:int(xtarget) + Npix / 2],
                         interpolation='nearest', cmap='viridis', origin='left')
        plt.colorbar(im2, orientation='horizontal', fraction=0.05, pad=0.1,
                     ticks=plt.MaxNLocator(5))

        residual_image = photobject.get_residual_image()
        ax3 = fig.add_subplot(1, 3, 3)
        im3 = ax3.imshow(
            residual_image[int(ytarget) - Npix / 2:int(ytarget) + Npix / 2,
                           int(xtarget) - Npix / 2:int(xtarget) + Npix / 2],
            interpolation='nearest', cmap='viridis', origin='left')
        plt.colorbar(im3, orientation='horizontal', fraction=0.05, pad=0.1,
                     ticks=plt.MaxNLocator(5))
        return groupmodel


class HSTPSFModel(object):
    """A psf model based on an HST image of a standard star or a 
    stack of stars."""

    def __init__(self, psfimfilename, targetpixscale, psfpixscale=0.03,
                 x_0=None, y_0=None, psf_recenter=False, fix_target_pos=True,
                 fwhm_arcsec=_HST_WFC3_PSF_FWHM_ARCSEC):
        """ 
        :param psfimfilename: full path for a .fits image file
        :param targetpixscale: arcseconds per pixel of the target image        
        :param psfpixscale: arcseconds per pixel of the psf image (if not 
        provided in the image header). Defaults to 0.03" per pixel
        :param xycenter: tuple, giving pixel coordinates of  the star. If not 
        provided, we assume it is at the center of the image.
        :param psf_recenter: execute a centroiding algorithm to locate the
        center of the psf
        :param fix_target_pos: execute "forced photometry" with the center of 
        the target fixed, and only the psf flux scaling as a free parameter.
        """
        psfimage = fits.open(psfimfilename)
        self.header = psfimage[0].header
        self.psfimdat = psfimage[0].data

        # TODO : check the header for pixel scale info

        if x_0 is None:
            self.xpsf = self.psfimdat.shape[0] / 2.,
            self.ypsf = self.psfimdat.shape[1] / 2.
        else:
            self.xpsf, self.ypsf = x_0, y_0

        if psf_recenter:
            self.xpsf, self.ypsf = centroid(self.psfimdat, x_0, y_0)

        self.psfmodel = photutils.psf.models.FittableImageModel(
            self.psfimdat, x_0=self.xpsf, y_0=self.ypsf,
            oversampling=targetpixscale / psfpixscale)

        # Fix the center of the psf for "forced photometry" -- no recentering
        if fix_target_pos:
            self.psfmodel.x_0.fixed = True
            self.psfmodel.y_0.fixed = True
        else:
            self.psfmodel.x_0.fixed = False
            self.psfmodel.y_0.fixed = False

        # Set up the grouper, background estimator, and fitter objects:
        self.grouper = DAOGroup(2.0 * fwhm_arcsec * psfpixscale)
        self.bkg_estimator = MMMBackground()
        self.fitter = LevMarLSQFitter()

        # Disabled b/c its unnecessary:
        # Measure the background noise level of the target image:
        # bkgrms = MADStdBackgroundRMS()
        # std = bkgrms(targetimdat)
        #
        # If we didn't know where the target was located, we could use the IRAF star
        # finder algorithm to locate sources.  In this case we won't actually
        # use a finder object, because we will feed in the known target location
        #  in the next step.
        #iraffind = IRAFStarFinder(threshold=15*std,
        #                        fwhm=sigma_psf_pixels*gaussian_sigma_to_fwhm,
        #                        minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
        #                        sharplo=0.0, sharphi=2.0)

