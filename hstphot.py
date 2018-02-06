#! /usr/bin/env python
# 2014.06.29  S.Rodney
__author__ = 'rodney'

import hstzpt_apcorr
import sys
import numpy as np
if sys.version_info <= (3,0):
    import exceptions


# Conversion table from full filter names to single-letter abbreviations
# used in STARDUST and other S.Rodney code
FilterAlpha = {'unknown': '?',
               'F225W': 'S', 'F275W': 'T', 'F336W': 'U', 'F390W': 'C',
               'F350LP': 'W',
               'F435W': 'B', 'F475W': 'G', 'F606W': 'V', 'F625W': 'R',
               'F775W': 'X', 'F814W': 'I', 'F850LP': 'Z',
               'F125W': 'J', 'F160W': 'H', 'F125W+F160W': 'A',
               'F105W': 'Y', 'F110W': 'M', 'F140W': 'N',
               'F098M': 'L', 'F127M': 'O', 'F139M': 'P', 'F153M': 'Q',
               'G141': '4', 'G102': '2', 'blank': '0',
               'F689M': '6', 'F763M': '7', 'F845M': '8',
               }


def getwcsobj(imfile_or_hdr, ext=0):
    """Create a WCS object from the header of the given image file.
    Checks if the WCS SIP distortion coefficients are included in the header,
    and removes them if redundant.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    drizzled = False
    fobj = None
    imfilename = ''
    if isinstance(imfile_or_hdr, str):
        fobj = fits.open(imfile_or_hdr)
        header = fits.getheader(imfile_or_hdr, ext=ext)
        imfilename = imfile_or_hdr.lower()
    elif isinstance(imfile_or_hdr, fits.Header):
        header = imfile_or_hdr
        if 'FILENAME' in header:
            imfilename = header['FILENAME']
    else:
        return None

    if imfilename.endswith('_drz.fits') or imfilename.endswith('_drc.fits'):
        drizzled = True

    # decide if this is a drizzled image file with SIP coefficients
    gotsip = False
    if not drizzled and 'DRIZCORR' in header:
        drizzled = header['DRIZCORR'].lower() == 'complete'
    if 'A_ORDER' in header:
        gotsip = True

    if drizzled and gotsip:
        for coeff in ['A','B']:
            for ix in range(header[coeff+'_ORDER']):
                for iy in range(header[coeff+'_ORDER']):
                    key = '%s_%i_%i'%(coeff, ix, iy)
                    if key in header:
                        header.remove(key)
            if coeff+'_ORDER' in header:
                header.remove(coeff+'_ORDER')

    try:
        wcs = WCS(fobj=fobj, header=header)
        if fobj is not None:
            fobj.close()
    except KeyError:
        wcs = WCS(header=header)
    return wcs


def radec2xy(imfile, ra, dec, ext=0):
    """ Convert the given ra,dec position (in decimal degrees) into
    x,y pixel coordinates on the given image.
    NOTE : the pixel values returned are in the fits style, with the center
      of the lower left pixel at (1,1).  The numpy/scipy convention sets
      the center of the lower left pixel at (0,0).
    """
    wcs = getwcsobj(imfile, ext=ext)
    x, y = wcs.wcs_world2pix(ra, dec, 1)
    return x, y


def xy2radec(imfile_or_hdr, x, y, ext=0):
    """ Convert the given x,y pixel position into
    ra,dec sky coordinates (in decimal degrees) for the given image.

    NOTE : this program assumes the input position follows the fits convention,
    with the center of the lower left pixel at (1,1).  The numpy/scipy
    convention sets the center of the lower left pixel at (0,0).

    :param imfile_or_hdr: image filename or astropy.io.fits Header object
    """
    wcs = getwcsobj(imfile_or_hdr, ext=ext)
    if wcs is None:
        print("WARNING: could not convert x,y to ra,dec for %s" %
              str(imfile_or_hdr))
        return 0, 0
    ra, dec = wcs.wcs_pix2world(x, y, 1)
    return ra, dec


def getxycenter(image, x, y, ext=0, radec=False,
                fitsconvention=False, verbose=False):
    """ Use a gaussian centroid algorithm to locate the center of a star
    near position x,y.

    :param image: any valid input to get_header_and_data(), namely:
      a string giving a fits filename, a pyfits hdulist or hdu, a pyfits
      primaryhdu object, a tuple or list giving [hdr,data]
    :param ext: (optional) extension number for science array
    :param radec: input coordinates are in RA,Dec instead of pixels
    :param fitsconvention: if True, then the input and output x,y pixel
        positions use the fits pixel indexing convention, with the center of
        the lower left pixel at (1,1). Otherwise this function uses the
        numpy/scipy convention which sets the center of the lower left pixel
        at (0,0).
    :param verbose: report progress verbosely
    :return: string giving the filter name as it appears in the fits header.
    """
    from PythonPhot import cntrd
    from astropy.io import fits as pyfits

    if radec:
        x, y = radec2xy(image, x, y, ext=ext)
    fwhmpix = getfwhmpix(image, ext=ext)
    imhdr, imdat = getheaderanddata(image, ext=ext)
    if fitsconvention:
        x, y = x - 1, y - 1
    xc, yc = cntrd.cntrd(imdat, x, y, fwhmpix, verbose=verbose)
    if xc == -1:
        if verbose:
            print('Recentering within a 5-pixel box')
        xc, yc = cntrd.cntrd(imdat, x, y, fwhmpix,
                             verbose=verbose, extendbox=5)
        if xc == -1: raise RuntimeError('Error : centroiding failed!')
    if fitsconvention:
        xc, yc = xc + 1, yc + 1
    return xc, yc


def getpixscale(image, returntuple=False, ext=None):
    """ Compute the pixel scale of the reference pixel in arcsec/pix in
    each direction from the fits header cd matrix.

    :param image: any valid input to getheader(), namely:
      a string giving a fits filename, a pyfits hdulist or hdu, a pyfits
      header object, a tuple or list giving [hdr,data]
    :param returntuple: bool. When True, return the two pixel scale values
         along the x and y axes.  When False, return the average of the two.
    :param ext: (optional) extension number for science array
    :return: float value or tuple giving the pixel scale in arcseconds/pixel.
    """
    from math import sqrt
    hdr = getheader(image, ext=ext)
    if 'CD1_1' in hdr:
        cd11 = hdr['CD1_1']
        cd12 = hdr['CD1_2']
        cd21 = hdr['CD2_1']
        cd22 = hdr['CD2_2']
        # define the sign based on determinant
        det = cd11 * cd22 - cd12 * cd21
        if det < 0:
            sgn = -1
        else:
            sgn = 1

        if cd12 == 0 and cd21 == 0:
            # no rotation: x=RA, y=Dec
            cdelt1 = cd11
            cdelt2 = cd22
        else:
            cdelt1 = sgn * sqrt(cd11 ** 2 + cd12 ** 2)
            cdelt2 = sqrt(cd22 ** 2 + cd21 ** 2)
    elif 'CDELT1' in hdr.keys() and \
            (hdr['CDELT1'] != 1 and hdr['CDELT2'] != 1):
        cdelt1 = hdr['CDELT1']
        cdelt2 = hdr['CDELT2']
    else:
        raise exceptions.RuntimeError(
            "Cannot identify CD matrix in %s" % image)
    cdelt1 *= 3600.
    cdelt2 *= 3600.
    if returntuple:
        return cdelt1, cdelt2
    else:
        return (abs(cdelt1) + abs(cdelt2)) / 2.


def getfwhmpix(image, ext=0):
    """ Determine the FWHM in pixels for this HST image.

    :param image: any valid input to getheader(), namely:
      a string giving a fits filename, a pyfits hdulist or hdu, a pyfits
      header object, a tuple or list giving [hdr,data]
    :param ext: (optional) extension number for science array
    :return: string giving the filter name as it appears in the fits header.
    """
    hdr = getheader(image, ext=ext)
    if ext != 0:
        hdr0 = getheader(image, ext=0)
    else:
        hdr0 = hdr
    camera = getcamera(hdr0)

    if camera == 'ACS-WFC':
        fwhmarcsec = 0.13
    elif camera == 'WFC3-UVIS':
        fwhmarcsec = 0.07
    elif camera == 'WFC3-IR':
        fwhmarcsec = 0.14
    elif ('TELESCOP' in hdr) and (hdr['TELESCOP'] == 'HST'):
        fwhmarcsec = 0.1
    else:
        print("WARNING : no instrument, detector or telescope identified."
              "  so we are arbitrarily setting the FWHM to 2.5 pixels for "
              "centroiding")
        return 2.5

    try:
        pixscale = getpixscale(hdr)
    except KeyError:
        print("WARNING : Can't determine the pixel scale from the header, "
              "  so we are arbitrarily setting the FWHM to 2.5 pixels for "
              "centroiding")
        return 2.5

    fwhmpix = fwhmarcsec / pixscale
    return fwhmpix


def getcamera(image):
    """ Determine the camera name (e.g. ACS-WFC or WFC3-IR)
    from the header of the given image.

    :param image: any valid input to getheader(), namely:
      a string giving a fits filename, a pyfits hdulist or hdu, a pyfits
      header object, a tuple or list giving [hdr,data]

    :return: string giving the dash-separated camera name
    """
    hdr = getheader(image)
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


def getfilter(image):
    """ Determine the filter name from the header of the given image.

    :param image: any valid input to getheader(), namely:
      a string giving a fits filename, a pyfits hdulist or hdu, a pyfits
      header object, a tuple or list giving [hdr,data]

    :return: string giving the filter name as it appears in the fits header.
    """
    hdr = getheader(image)
    if 'FILTER' in hdr:
        return hdr['FILTER']
    elif 'FILTER1' in hdr:
        if hdr['FILTER1'].startswith('CLEAR'):
            return hdr['FILTER2']
        else:
            return hdr['FILTER1']


def getheader(fitsfile, ext=None):
    """ Return a fits image header.

    :param image: a string giving a fits filename, a pyfits hdulist or hdu,
      a pyfits header object, a tuple or list giving [hdr,data]

    :return: the pyfits header object
    """
    from astropy.io import fits as pyfits

    if isinstance(fitsfile, (tuple, list)):
        hdr, data = fitsfile
    else:
        if isinstance(fitsfile, str):
            fitsfile = pyfits.open(fitsfile)
        if isinstance(fitsfile, pyfits.header.Header):
            hdr = fitsfile
        elif isinstance(fitsfile, pyfits.hdu.hdulist.HDUList):
            if ext is not None:
                hdr = fitsfile[ext].header
            else:
                extnamelist = [hdu.name.lower() for hdu in fitsfile]
                if 'sci' in extnamelist:
                    isci = extnamelist.index('sci')
                    hdr = fitsfile[isci].header
                else:
                    hdr = fitsfile[0].header
        elif isinstance(fitsfile, pyfits.hdu.image.PrimaryHDU):
            hdr = fitsfile.header
        else:
            raise exceptions.RuntimeError('input object type %s is unrecognized')
    return hdr


def getheaderanddata(image, ext=None):
    """ Return a fits image header and data array.

    :param image: a string giving a fits filename, a pyfits hdulist or hdu,
      a pyfits primaryhdu object, a tuple or list giving [hdr,data]
    :param ext: (optional) extension number for science array
    :return:  pyfits.header.Header object and numpy data array
    """
    from astropy.io import fits as pyfits
    if isinstance(image, str):
        imfilename = image
        image = pyfits.open(imfilename)
    if isinstance(image, pyfits.hdu.hdulist.HDUList):
        if ext is not None:
            hdr = image[ext].header
            data = image[ext].data
        else:
            extnamelist = [hdu.name.lower() for hdu in image]
            if 'sci' in extnamelist:
                isci = extnamelist.index('sci')
                hdr = image[isci].header
                data = image[isci].data
            else:
                hdr = image[0].header
                data = image[0].data
    elif isinstance(image, pyfits.hdu.image.PrimaryHDU):
        hdr = image.header
        data = image.data
    elif isinstance(image, (tuple, list)):
        if len(image)==2:
            hdr, data = image
        else:
            raise exceptions.RuntimeError('Input list/tuple must have exactly'
                                          ' 2 entries, giving [hdr,data]')
    else:
        raise exceptions.RuntimeError('input object type %s is unrecognized')

    return hdr, data


def doastropyphot(targetimfilename, xy, psfimfilename=None,
                  psfpixscale=None, recenter_target=True,
                  apradarcsec=[0.1,0.2,0.3], skyannradarcsec=[3.0,5.0],
                  fitpix=11, targetname='TARGET',
                  ntestpositions=100, psfradpix=3,
                  skyannpix=None, skyalgorithm='sigmaclipping',
                  setskyval=None, recenter_fakes=True,
                  exptime=1, exact=True, ronoise=1, phpadu=1, verbose=False,
                  debug=False):
    """ Measure the flux (and uncertainty?) using the astropy-affiliated
    photutils package.

    :param targetimfilename: name of the .fits image file with the target (the
      star to be photometered).
    :param xy:
    :param psfimfilename: name of the .fits image file with the PSF image (the
      star that defines a PSF model to be fit to the target)
    :param psfpixscale: Pixel scale of the PSF star (necessary if header of the
      PSF star image does not provide WCS keywords that define the pixel scale)
    :param recenter_target: boolean;  Use a centroiding algorithm to locate the
      center of the target star. Set to 'False' for "forced photometry"
    :param apradpix: list of aperture radii in arcsec, for aperture photometry
    :param skyannradarcsec: inner and outer radii in arcsec for the sky annulus
      (the annulus in which the sky is measured)

    :param ntestpositions:
    :param psfradpix:
    :param skyalgorithm:
    :param setskyval:
    :param recenter_fakes:
    :param exptime:
    :param exact:
    :param ronoise:
    :param phpadu:
    :param verbose:
    :param debug:
    :return:
    """
    from . import astropyphot
    from os import path

    targetim = astropyphot.TargetImage(targetimfilename)
    if psfpixscale is None:
        psfpixscale = targetim.pixscale
    targetim.set_target(x_0=xy[0], y_0=xy[1], targetname=targetname,
                        recenter=recenter_target)

    targetim.doapphot(apradarcsec, units='arcsec')
    # apphotresults = targetim.photometry['aperturephot'].phot_table

    if psfimfilename is not None:
        psfmodelname = path.basename(psfimfilename)
        targetim.load_psfmodel(psfimfilename, psfmodelname,
                               psfpixscale=psfpixscale)
        apradpix = np.max([5, np.round(fitpix/2.)])
        targetim.dopsfphot(psfmodelname, fitpix=fitpix, apradpix=apradpix)
        # psfphotresults = targetim.photometry[psfmodelname].phot_table

    return targetim

    # TODO : extract flux and error from aper and psf photometry!!
    #psfflux =psfphotresults['']
    #psffluxerr = 0.0

    # TODO : convert astropyphot results into a string for printing
    # return(apphotresults, psfphotresults)
    # return apflux, apfluxerr, psfflux, psffluxerr, sky, skyerr

    # the sky measurement algorithm in photutils is not quite the same as
    # in PythonPhot, so for now, we adopt the PythonPhot sky value instead
    # of letting photutils compute it.
    #if skyval is None:
    #    skyval = sky
    # output_photutils = astropyphot.get(
    #     imdat, psfimage, [xpy, ypy],
    #     psfradpix=psfradpix, apradpix=appix, ntestpositions=ntestpositions,
    #     skyannpix=skyannpix, skyalgorithm=skyalgorithm, setskyval=skyval,
    #     recenter_target=recenter, recenter_fakes=True, exact=exact,
    #     exptime=exptime, ronoise=1, phpadu=phpadu, verbose=verbose,
    #     debug=debug)
    # apflux, apfluxerr, psfflux, psffluxerr, sky, skyerr = output_photutils


def dopythonphot(image, xc, yc, aparcsec=0.4, system='AB', ext=None,
                 psfimage=None, psfradpix=3, recenter=False, imfilename=None,
                 ntestpositions=100, snthresh=0.0, zeropoint=None,
                 filtername=None, exptime=None, pixscale=None,
                 skyannarcsec=[6.0, 12.0], skyval=None,
                 skyalgorithm='sigmaclipping',
                 target=None, printstyle=None, exact=True, fitsconvention=True,
                 phpadu=None, returnflux=False, showfit=False,
                 verbose=False, debug=False):
    """ Measure the flux through aperture(s) and/or psf fitting using the
    PythonPhot package.

    Inputs:
      image :  string giving image file name OR a list or 2-tuple giving
               the header and data array as  [hdr,data]
      xc,yc  :  aperture center in pixel coordinates
      aparcsec : aperture radius in arcsec, or a string with a comma-separated
                 list of aperture radii
      psfimage : filename for a fits file containing a psf model
      system :  report AB or Vega mags ('AB','Vega')
      snthresh :  If the measured flux is below <snthresh>*fluxerr then the
                resulting magnitude is reported as a lower limit.
      zeropoint : fix the zeropoint (if not provided, we look it up from
      hardcoded tables)
      skyannarcsec : inner and outer radius of the sky annulus (in arcsec)
      target : name of the target object (for printing in snanastyle)
      printstyle :  None or 'default' = report MJD, filter, and photometry
                    'verbose' or 'long' = include target name and position
                    'snana' = report mags in the format of a SNANA .dat file.
      fitsconvention : xc,yc position follows the fits convention with (1,1)
             as the lower left pixel.  Otherwise, follow the python/pyfits
             convention with (0,0) as the lower left pixel.
      returnflux : instead of returning a list of strings containing all the
             flux and magnitude information, simply return a single flux val
    Note :  No recentering is done (i.e. this does forced photometry at the
       given pixel position)
    """
    from PythonPhot import photfunctions

    if debug == 1:
        import pdb
        pdb.set_trace()

    imhdr, imdat = getheaderanddata(image, ext=ext)
    if imfilename is None:
        if isinstance(image, str):
            imfilename = image
        elif 'FILENAME' in imhdr:
            imfilename = imhdr['FILENAME']
        else:
            imfilename = 'unknown'

    if imdat.dtype != 'float64':
        imdat = imdat.astype('float64', copy=False)
    if not filtername:
        if 'FILTER1' in imhdr:
            if 'CLEAR' in imhdr['FILTER1']:
                filtername = imhdr['FILTER2']
            else:
                filtername = imhdr['FILTER1']
        else:
            filtername = imhdr['FILTER']

    if not exptime:
        if 'EXPTIME' in imhdr:
            exptime = imhdr['EXPTIME']
        else:
            raise exceptions.RuntimeError(
                "Cannot determine exposure time for %s" % imfilename)

    if not pixscale:
        pixscale = getpixscale(imhdr, ext=ext)
        if not np.iterable(aparcsec):
            aparcsec = np.array([aparcsec])
        elif not isinstance(aparcsec, np.ndarray):
            aparcsec = np.array(aparcsec)

    appix = np.array([ap / pixscale for ap in aparcsec])
    skyannpix = np.array([skyrad / pixscale for skyrad in skyannarcsec])
    if len(appix) >= 1:
        assert skyannpix[0] >= np.max(
            appix), "Sky annulus must be >= largest aperture."
    camera = getcamera(imhdr)

    # Define the conversion factor from the values in this image
    # to photons : photons per ADU.
    if phpadu is None:
        if 'BUNIT' not in imhdr:
            if camera == 'WFC3-IR' and 'EXPTIME' in imhdr:
                phpadu = imhdr['EXPTIME']
            else:
                phpadu = 1
        elif imhdr['BUNIT'].lower() in ['cps', 'electrons/s']:
            phpadu = imhdr['EXPTIME']
        elif imhdr['BUNIT'].lower() in ['counts', 'electrons']:
            phpadu = 1
        assert (
            phpadu is not None), "Can't determine units from the image header."

    if fitsconvention:
        xpy, ypy = xc - 1, yc - 1
    else:
        xpy, ypy = xc, yc

    if recenter:
        xim, yim = getxycenter([imhdr, imdat], xc, yc,
                               fitsconvention=True, radec=False,
                               verbose=verbose)
        if verbose:
            print("Recentered position (x,y) : %.2f %.2f" % (xim, yim))
            ra, dec = xy2radec(imhdr, xim, yim)
            print("Recentered position (ra,dec) : %.6f %.6f" % (ra, dec))

    output_PythonPhot = photfunctions.get_flux_and_err(
        imdat, psfimage, [xpy, ypy],
        psfradpix=psfradpix, apradpix=appix, ntestpositions=ntestpositions,
        skyannpix=skyannpix, skyalgorithm=skyalgorithm, setskyval=skyval,
        recenter_target=False, recenter_fakes=True, exact=exact,
        exptime=exptime, ronoise=1, phpadu=phpadu,
        showfit=showfit, verbose=verbose, debug=debug)
    apflux, apfluxerr, psfflux, psffluxerr, sky, skyerr = output_PythonPhot

    if not np.iterable(apflux):
        apflux = np.array([apflux])
        apfluxerr = np.array([apfluxerr])

    # Define aperture corrections for each aperture
    if zeropoint is not None:
        zpt = zeropoint
        apcor = np.zeros(len(aparcsec))
        aperr = np.zeros(len(aparcsec))
    else:
        zpt = hstzpt_apcorr.getzpt(image, system=system)
        if camera == 'WFC3-IR':
            # TODO: allow user to choose an alternate EE table?
            apcor, aperr = hstzpt_apcorr.apcorrWFC3IR(filtername, aparcsec)
        elif camera == 'WFC3-UVIS':
            apcor, aperr = hstzpt_apcorr.apcorrWFC3UVIS(filtername, aparcsec)
        elif camera == 'ACS-WFC':
            apcor, aperr = hstzpt_apcorr.apcorrACSWFC(filtername, aparcsec)

    # record the psf flux as a final infinite aperture for printing purposes
    if psfimage is not None:
        aparcsec = np.append(aparcsec, np.inf)
        apflux = np.append(apflux, [psfflux])
        apfluxerr = np.append(apfluxerr, [psffluxerr])
        apcor = np.append(apcor, 0)

    # apply aperture corrections to flux and mags
    # and define upper limit mags for fluxes with significance <snthresh
    mag, magerr = np.zeros(len(apflux)), np.zeros(len(apflux))
    for i in range(len(apflux)):
        if np.isfinite(aparcsec[i]):
            # For actual aperture measurements (not the psf fitting flux),
            # apply aperture corrections to the measured fluxes
            # Flux rescaled to larger aperture:
            apflux[i] *= 10 ** (0.4 * apcor[i])
            # Flux error rescaled:
            df = apfluxerr[i] * 10 ** (0.4 * apcor[i])
            #  Systematic err from aperture correction :
            dfap = 0.4 * np.log(10) * apflux[i] * aperr[i]
            apfluxerr[i] = np.sqrt(df ** 2 + dfap ** 2)  # total flux err
            if verbose > 1:
                print(" FERRTOT  FERRSTAT   FERRSYS")
                print(" %.5f  %.5f  %.5f" % (apfluxerr[i], df, dfap))

        if apflux[i] < abs(apfluxerr[i]) * snthresh:
            # no real detection. Report mag as an upper limit
            sigmafactor = snthresh or 3
            mag[i] = -2.5 * np.log10(sigmafactor * abs(apfluxerr[i])) \
                     + zpt - apcor[i]
            magerr[i] = -9.0
        else:
            # Good detection. convert to a magnitude (ap correction already
            # applied)
            mag[i] = -2.5 * np.log10(apflux[i]) + zpt
            magerr[i] = 1.0857 * apfluxerr[i] / apflux[i]

    if debug:
        import pdb
        pdb.set_trace()

    if returnflux:
        return apflux

    if 'EXPSTART' in imhdr and 'EXPEND' in imhdr:
        mjdobs = (imhdr['EXPEND'] + imhdr['EXPSTART'])/2.
    else:
        mjdobs = 0.0

    if verbose and printstyle == 'snana':
        # Convert to SNANA fluxcal units and Construct a SNANA-style OBS
        # line, e.g.
        # OBS: 56456.500  H  wol    0.000    8.630   25.160   -9.000
        fluxcal = apflux * 10 ** (0.4 * (27.5 - zpt))
        fluxcalerr = apfluxerr * 10 ** (0.4 * (27.5 - zpt))
        print('VARLIST:  MJD  FLT FIELD   FLUXCAL   FLUXCALERR    MAG     '
              'MAGERR   ZPT')
    elif verbose:
        if printstyle.lower() in ['long', 'verbose']:
            print('#  TARGET                RA         DEC       MJD  FILTER '
                  ' APER       FLUX  FLUXERR         MAG   MAGERR  MAGSYS    '
                  '   ZP      SKY SKYERR  IMAGE')
        else:
            print('# MJD     FILTER  APER      FLUX   FLUXERR       MAG     '
                  'MAGERR  MAGSYS    ZP       SKY   SKYERR')

    if printstyle is not None:
        printstyle = printstyle.lower()
    ra, dec = 0, 0
    if (printstyle is not None and
                printstyle.lower() in ['snana', 'long', 'verbose']):
        if not target and 'FILENAME' in imhdr.keys():
            target = imhdr['FILENAME'].split('_')[0]
        elif not target:
            target = 'target'
        ra, dec = xy2radec(imhdr, xc, yc, ext=ext)

    maglinelist = []
    for iap in range(len(aparcsec)):
        if printstyle == 'snana':
            magline = 'OBS: %8.2f   %6s   %s %8.3f %8.3f    '\
                      '%8.3f %8.3f   %.3f' % (
                          float(mjdobs), FilterAlpha[filtername], target,
                          fluxcal[iap], fluxcalerr[iap], mag[iap], magerr[iap],
                          zpt)
        elif printstyle in ['long', 'verbose']:
            magline = '%-15s  %10.5f  %10.5f  %.3f  %6s  %4.2f  %9.4f %8.4f  '\
                      ' %9.4f %8.4f  %5s   %7.4f  %7.4f %6.4f  %s' % (
                          target, ra, dec, float(mjdobs), filtername,
                          aparcsec[iap],
                          apflux[iap], apfluxerr[iap], mag[iap], magerr[iap],
                          system,
                          zpt, sky, skyerr, imfilename)
        else:
            magline = '%.3f  %6s  %4.2f  %9.4f %8.4f   %9.4f %8.4f  %5s   ' \
                      '%7.4f  %7.4f %6.4f' % (
                          float(mjdobs), filtername, aparcsec[iap],
                          apflux[iap], apfluxerr[iap], mag[iap], magerr[iap],
                          system,
                          zpt, sky, skyerr)
        maglinelist.append(magline)

    return maglinelist


def main():
    import os
    import argparse
    from astropy.io import fits as pyfits

    parser = argparse.ArgumentParser(
        description='Measure aperture and/or PSF photometry on drizzled HST '
                    ' images using either the PythonPhot routines or the '
                    'astropy-affiliated photutils package.')

    # Required positional argument
    parser.add_argument('image', help='Drizzled HST image fits file.')
    parser.add_argument('x', type=float, help='X position or R.A.')
    parser.add_argument('y', type=float, help='Y position or Dec')
    parser.add_argument('--photpackage', type=str, default='PythonPhot',
                        choices=['PythonPhot', 'photutils'],
                        help="Underlying photometry package to use.")
    parser.add_argument('--psfmodel', type=str, default=None,
                        help="Filename of a psf model fits file.")
    parser.add_argument('--ntest', type=int, default=100,
                        help='Number of test positions for fake sources.')
    parser.add_argument('--ext', type=int, default=None,
                        help='Specify the fits extension number. Required '
                             'for FLT files.')
    parser.add_argument('--forced', action='store_true',
                        help='Forced photometry mode (no recentering).')
    parser.add_argument('--radec', action='store_true',
                        help='x,y give RA and Dec instead of pixel position.')
    parser.add_argument('--AB', action='store_true',
                        help='Report AB mags (the default).')
    parser.add_argument('--vega', action='store_true',
                        help='Report Vega mags.')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Compute sub-pixel areas approximately (Faster. '
                             'Safe for apertures much larger than a pixel.)')
    # TODO: allow user to skip aperture photometry with PythonPhot
    parser.add_argument('--apertures', type=str, default='0.4',
                        help='Size of photometry aperture(s) in arcsec. ')
    parser.add_argument('--filtername', type=str, default=None,
                        help='HST filter name (if not provided, we will read '
                             'it from the header)')
    parser.add_argument('--pixscale', type=float, default=None,
                        help='Pixel scale (arcsec/pixel) -- if not provided, '
                             'we will try to read it from the header)')
    parser.add_argument('--exptime', type=float, default=None,
                        help='Exposure time in sec (if not provided, we will '
                             'try to read it from the header)')
    parser.add_argument('--skyannulus', metavar='6.0,12.0', type=str,
                        default='6.0,12.0',
                        help='Inner and outer radius of sky annulus in '
                             'arcsec. ')
    parser.add_argument('--skyval', type=float, default=None,
                        help='Fix the sky flux per pixel to this value.')
    parser.add_argument('--skyalgorithm', choices=['sigmaclipping', 'mmm'],
                        default='sigmaclipping',
                        help='Set the algorithm to use for measuring the sky.')
    parser.add_argument('--zeropoint', type=float, default=None,
                        help='Fix the zero point. If not provided, we use a '
                             'hardcoded table for HST filters.')
    parser.add_argument('--snthresh', type=float, default=0,
                        help='When the S/N is below this threshold, '
                             'the magnitude is printed as an N-sigma lower '
                             'limit.')
    parser.add_argument('--printstyle', type=str,
                        choices=['snana', 'long', 'short'], default='short',
                        help='Report photometry in long format or as a '
                             'SNANA-style "OBS:" line.')
    parser.add_argument('--target', type=str, default=None,
                        help="Name of target (for 'long' and 'snana' print "
                             "styles)")
    parser.add_argument('--phpadu', type=float, default=None,
                        help='Photons per ADU (for converting data numbers '
                             'to photon counts).')
    parser.add_argument('--showfit', action='store_true',
                        help='Show the target, scaled PSF model '
                             'and residual images.')
    parser.add_argument('-v', dest='verbose', action='count', default=0,
                        help='Turn verbosity up (use -v,-vv,-vvv, etc.)')
    parser.add_argument('-d', dest='debug', action='count', default=0,
                        help='Turn up debugging depth (use -d,-dd,-ddd)')

    # TODO: allow user to choose an alternate EE table?

    argv = parser.parse_args()

    # Allow the user to specify the fits extension number in brackets
    # after the image name. e.g.  some_image_flt.fits[1]
    if argv.ext is None and argv.image.endswith(']'):
        argv.image, argv.ext = os.path.basename(argv.image).split('[')
        argv.ext = int(argv.ext.rstrip(']'))

    hdr = pyfits.getheader(argv.image)
    if argv.image.endswith('flt.fits') or argv.image.endswith('flc.fits'):
        if argv.ext is None:
            raise exceptions.RuntimeError(
                'For FLT files you must specify the fits extension number.')
    if 'NEXTEND' in hdr:
        if hdr['NEXTEND'] > 1 and argv.ext is None:
            raise exceptions.RuntimeError(
                'For MEF files you must specify the fits extension number.')

    if argv.radec:
        xim, yim = radec2xy(argv.image, argv.x, argv.y, ext=argv.ext)
        if int(argv.verbose or 0) > 1:
            print("x,y position [fits-style, (1,1)-indexed]: %.2f %.2f" % (
                xim, yim))
    else:
        xim, yim = argv.x, argv.y

    magsys = 'AB'
    if argv.vega:
        magsys = 'Vega'
    if argv.AB:
        magsys = 'AB'

    if argv.apertures is not None:
        aplist = np.array([float(ap) for ap in argv.apertures.split(',')])
    else:
        aplist = []
    skyannarcsec = [float(ap) for ap in argv.skyannulus.split(',')]
    if argv.photpackage.lower() == 'pythonphot':
        maglinelist = dopythonphot(
            argv.image, xim, yim, aplist, system=magsys,
            psfimage=argv.psfmodel, ext=argv.ext,
            skyannarcsec=skyannarcsec, skyval=argv.skyval,
            zeropoint=argv.zeropoint,
            filtername=argv.filtername, exptime=argv.exptime,
            pixscale=argv.pixscale, skyalgorithm=argv.skyalgorithm,
            snthresh=argv.snthresh, exact=(not argv.fast),
            ntestpositions=argv.ntest,
            recenter=not argv.forced,
            printstyle=argv.printstyle, target=argv.target,
            phpadu=argv.phpadu, showfit=argv.showfit,
            verbose=argv.verbose, debug=argv.debug)
        for iap in range(len(maglinelist)):
            print(maglinelist[iap].strip())

if __name__ == '__main__':
    main()
