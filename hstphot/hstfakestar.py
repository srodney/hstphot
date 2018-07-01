__author__ = 'rodney'

from .__main__ import radec2xy
from astropy.io import fits as pyfits
import numpy as np
from PythonPhot.photfunctions import rdpsfmodel
from PythonPhot.photfunctions import addtoimarray

def addtofits(fitsin, fitsout, psfmodelfile, position, fluxscale,
              coordsys='xy', verbose=False):
    """ Create a psf, add it into the given fits image at the

    :param fitsin: fits file name for the target image
    :param psfmodelfile: fits file holding the psf model
    :param position: Scalar or array, giving the psf center positions
    :param fluxscale: Scalar or array, giving the psf flux scaling factors
    :param coordsys: Either 'radec' or 'xy'
    :return:
    """

    if not np.iterable(fluxscale):
        fluxscale = np.array([fluxscale])
        position = np.array([position])

    if coordsys == 'radec':
        ra = position[:, 0]
        dec = position[:, 1]
        position = radec2xy(fitsin, ra, dec, ext=0)

    image = pyfits.open(fitsin, mode='readonly')
    imagedat = image[0].data

    psfmodel = rdpsfmodel(psfmodelfile)

    for i in range(len(fluxscale)):
        imagedat = addtoimarray(imagedat, psfmodel, position[i], fluxscale[i])
    image[0].data = imagedat

    image.writeto(fitsout, clobber=True)
    if verbose:
        print("Wrote updated image to %s" % fitsout)
    return
