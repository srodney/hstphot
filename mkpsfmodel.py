#! /usr/bin/env python
import os
import numpy as np
from PythonPhot import getpsf, aper
from astropy.io import fits as pyfits


def getcamera(imfile):
    """ Determine the camera name (e.g. ACS-WFC or WFC3-IR)
    from the header of the given image.  The input imfile
    may be a string giving a fits filename, a pyfits hdu object,
    or a pyfits header.
    :param imfile:
    :return:
    """
    import hstphot
    hdr = hstphot.getheader(imfile)
    instrument, detector = '', ''
    if 'CAMERA' in hdr:
        instrument = hdr['CAMERA']
        detector = ''
    elif 'INSTRUME' in hdr:
        instrument = hdr['INSTRUME']
    if 'DETECTOR' in hdr:
        detector = hdr['DETECTOR']
    camera = '-'.join([instrument, detector]).rstrip('-')
    return camera


def mkpsfmodel_stdstar(psfimdir="./", psfrad=0.3, fitrad=0.15,
                       starname='g191b2b',
                       pixscale=0.03,
                       bandlist=['f435w', 'f606w', 'f814w'],
                       verbose=True):
    """ Construct a set of psf models from drizzled HST images of a
    standard star, with the star in the center of the image.

    :param psfimdir: psf image directory (where the input std star images are
       and also where the output psf model images will be put)
    :param psfrad: the scalar radius, in arcsec, of the circular area within
                which the PSF will be defined.  This should be slightly larger
                than the radius of the brightest star that one will be
                interested in.
    :param fitrad: the scalar radius, in arcsec, of the circular area used in
               the least-square star fits.  Stetson suggest that fitrad should
               approximately equal to the FWHM, slightly less for crowded
               fields.  (fitrad must be smaller than psfrad.)

    :return : None
    """
    # radius of aperture and sky annulus for first-pass aperture photometry,
    # used to scale the psf
    aparcsec = 2.5
    skyradarcsec = np.array([2.5, 3.5])

    for band in bandlist:

        # convert user-supplied radii from arcsec to pixels
        appix = aparcsec / pixscale
        skyradpix = skyradarcsec / pixscale
        psfradpix = psfrad / pixscale
        fitradpix = fitrad / pixscale

        inputfile = os.path.join(
            psfimdir, '%s.e00/%s_%s_e00_reg_drz_sci.fits' % (
                starname, starname, band))
        outputfile = os.path.join(
            psfimdir, '%s_%s_%2imas_psf_model.fits' % (
                starname, band, int(pixscale * 1000)))

        hdulist = pyfits.open(inputfile)
        hdr = hdulist[0].header
        imdat = hdulist[0].data
        if 'FILTER1' in hdr:
            if 'CLEAR' in hdr['FILTER1']:
                filtname = hdr['FILTER2']
            else:
                filtname = hdr['FILTER1']
        else:
            filtname = hdr['FILTER']
        camera = getcamera(hdr)

        # Define the conversion factor from the values in this image
        # to photons : photons per ADU.
        if 'BUNIT' not in hdr:
            if camera == 'WFC3-IR' and 'EXPTIME' in hdr:
                phpadu = hdr['EXPTIME']
            else:
                phpadu = 1
        elif hdr['BUNIT'].lower() in ['cps', 'electrons/s']:
            phpadu = hdr['EXPTIME']
        elif hdr['BUNIT'].lower() in ['counts', 'electrons']:
            phpadu = 1
        assert (phpadu is not None), \
            "Can't determine units from the image header."

        rdnoise = 0
        if 'READNSEA' in hdr:
            rdnoise = np.mean([hdr[key] for key in hdr.keys()
                               if key.startswith('READNSE')])

        xmid = np.array([hdr['NAXIS1'] / 2.]) - 1
        ymid = np.array([hdr['NAXIS2'] / 2.]) - 1
        xpos, ypos = hstphot.getxycenter(
            inputfile, xmid, ymid, ext=0, radec=False,
            fitsconvention=False, verbose=True)
        if verbose:
            print("PSF recentering : (%.2f,%.2f) ==> (%.2f,%.2f)" % (
                xmid, ymid, xpos, ypos))
        xpos = np.array([xpos])
        ypos = np.array([ypos])

        idpsf = np.arange(len(xpos))
        image = pyfits.getdata(inputfile)

        # run aper to get mags and sky values for specified coords
        mag, magerr, flux, fluxerr, sky, skyerr, badflag, outstr = \
            aper.aper(image, xpos, ypos, phpadu=phpadu, apr=appix,
                      zeropoint=25,
                      skyrad=skyradpix, badpix=[-12000, 60000], exact=True)

        # use the star at those coords to generate a PSF model
        gauss, psf, psfmag = getpsf.getpsf(image, xpos, ypos, mag,
                                           np.asfarray([sky]), rdnoise,
                                           phpadu,
                                           idpsf, psfradpix, fitradpix,
                                           outputfile, zeropoint=25,
                                           debug=False)
        if verbose:
            print("PSF image written to %s" % outputfile)
    return


def mkpsfmodel(psfimage, psfrad=0.6, fitrad=0.3, pixscale=0.03,
               phpadu=1, rdnoise=0, mag=25, zeropoint=25, sky=0,
               verbose=True):
    """ Construct a psf model from drizzled HST images of a
    standard star or composite star, with the star in the center of the image.

    :param psfimage: filename of the .fits file with the star at the center in
               image array 0.
    :param psfrad: the scalar radius, in arcsec, of the circular area within
                which the PSF will be defined.  This should be slightly larger
                than the radius of the brightest star that one will be
                interested in measuring photometry for.
    :param fitrad: the scalar radius, in arcsec, of the circular area used in
               the least-square star fits.  Stetson suggest that fitrad should
               approximately equal to the FWHM, slightly less for crowded
               fields.  (fitrad must be smaller than psfrad.)
    :param pixscale: the pixel scale of the input image [arcseconds per pixel]
    :param phpadu:  the "gain" of the input image in photons per ADU  [?]
    :param rdnoise: the readnoise of the input image, in ADU [?]
    :param mag: the magnitude of the star in the input image
    :param zeropoint: the zero point of the input image (magnitude that
             produces a flux of 1 [ADU per second?]
    :param sky: the sky brightness of the input image, in ADU per second [?]
    :return : None
    """
    # TODO : currently the units in the __doc__ text are guesses. Need to
    # review the code  in the PythonPhot source and check
    import hstphot

    # convert user-supplied radii from arcsec to pixels
    psfradpix = psfrad / pixscale
    fitradpix = fitrad / pixscale

    outputfile = psfimage.replace('.fits', '_model.fits')
    hdulist = pyfits.open(psfimage)
    hdr = hdulist[0].header
    imdat = hdulist[0].data

    xmid = np.array([hdr['NAXIS1'] / 2.]) - 1
    ymid = np.array([hdr['NAXIS2'] / 2.]) - 1
    xpos, ypos = hstphot.getxycenter(
        psfimage, xmid, ymid, ext=0, radec=False,
        fitsconvention=False, verbose=True)
    if verbose:
        print("PSF recentering : (%.2f,%.2f) ==> (%.2f,%.2f)" % (
            xmid, ymid, xpos, ypos))
    xpos = np.array([xpos])
    ypos = np.array([ypos])
    mag = np.array([mag])
    idpsf = np.arange(len(xpos))

    # use the star at those coords to generate a PSF model
    gauss, psf, psfmag = getpsf.getpsf(
        imdat, xpos, ypos, mag,  np.asfarray([sky]), rdnoise,
        phpadu, idpsf, psfradpix, fitradpix,  outputfile, zeropoint=zeropoint,
        debug=False)
    if verbose:
        print("PSF image written to %s" % outputfile)
    return outputfile, gauss, psf, psfmag


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Make a PythonPhot psf model from the a single-star image")

    # Required positional argument
    parser.add_argument('inputimage',
                        help='FITS file with a single star at the center.')

    # optional keyword arguments
    parser.add_argument('--pixscale', type=float,
                        default=0.03,
                        help="arcseconds per pixel.")

    argv = parser.parse_args()

    mkpsfmodel(argv.inputimage, pixscale=argv.pixscale)


if __name__ == '__main__':
    main()
