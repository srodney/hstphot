#! /usr/bin/env python
# 2014.06.29  S.Rodney
__author__ = 'rodney'

import exceptions
import sys

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


def radec2xy(imfile, ra, dec, ext=0):
    """ Convert the given ra,dec position (in decimal degrees) into
    x,y pixel coordinates on the given image.
    NOTE : the pixel values returned are in the fits style, with the center
      of the lower left pixel at (1,1).  The numpy/scipy convention sets
      the center of the lower left pixel at (0,0).
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    fobj = fits.open(imfile)
    header = fits.getheader(imfile, ext=ext)
    try:
        wcs = WCS(fobj=fobj, header=header)
    except KeyError:
        wcs = WCS(header=header)
    fobj.close()
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
    from astropy.io import fits
    from astropy.wcs import WCS

    if isinstance(imfile_or_hdr, basestring):
        header = fits.getheader(imfile_or_hdr, ext=ext)
    elif isinstance(imfile_or_hdr, fits.Header):
        header = imfile_or_hdr
    else:
        print("WARNING: could not convert x,y to ra,dec for %s" %
               str(imfile_or_hdr))
    # try:
    # alternate WCS construction may be necessary for ACS files ?
    # wcs = WCS(fobj=fobj, header=header)
    # except KeyError:
    wcs = WCS(header=header)
    # fobj.close()
    ra, dec = wcs.all_pix2world(x, y, 1)
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
    imdat = pyfits.getdata(image, ext=ext)
    if fitsconvention:
        x, y = x - 1, y - 1
    xc, yc = cntrd.cntrd(imdat, x, y, fwhmpix, verbose=verbose)
    if xc == -1:
        if verbose:
            print('Recentering within a 5-pixel box')
        xc, yc = cntrd.cntrd(imdat, x, y, fwhmpix,
                             verbose=verbose, extendbox=5)
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
        print "WARNING : no instrument, detector or telescope identified."
        "  so we are arbitrarily setting the FWHM to 2.5 pixels for "
        "centroiding"
        return 2.5
    pixscale = getpixscale(hdr)
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
        if isinstance(fitsfile, basestring):
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


def get_header_and_data(image, ext=None):
    """ Return a fits image header and data array.

    :param image: a string giving a fits filename, a pyfits hdulist or hdu,
      a pyfits primaryhdu object, a tuple or list giving [hdr,data]
    :param ext: (optional) extension number for science array
    :return:  pyfits.header.Header object and numpy data array
    """
    from astropy.io import fits as pyfits
    if isinstance(image, basestring):
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





# Average flux of vega through an infinite aperture for ACS filters :
FLUX_VEGA_ACS_WFC_INF = {'F435W': 6.384e-09, 'F475W': 5.290e-09,
                         'F502N': 4.660e-09,
                         'F550M': 3.416e-09, 'F555W': 3.811e-09,
                         'F606W': 2.869e-09,
                         'F625W': 2.351e-09, 'F658N': 1.776e-09,
                         'F660N': 1.932e-09,
                         'F775W': 1.287e-09, 'F814W': 1.134e-09,
                         'F850LP': 8.261e-10,
                         'F892N': 8.729e-10}

# Here are the infinite aperture Vegamag zeropoints
ZPT_WFC3_IR_VEGA = {'F105W': 25.6236, 'F110W': 26.0628, 'F125W': 25.3293,
                    'F140W': 25.3761, 'F160W': 24.6949, 'F098M': 25.1057,
                    'F127M': 23.6799, 'F139M': 23.4006, 'F153M': 23.2098,
                    'F126N': 21.9396, 'F128N': 21.9355, 'F130N': 22.0138,
                    'F132N': 21.9499, 'F164N': 21.5239, 'F167N': 21.5948}

ZPT_WFC3_UVIS_VEGA = {'F200LP': 26.8902, 'F300X': 23.5363, 'F350LP': 26.7874,
                      'F475X': 26.2082, 'F850LP': 25.5505, 'F600LP': 23.3130,
                      'F218W': 21.2743, 'F225W': 22.3808, 'F275W': 22.6322,
                      'F336W': 23.4836, 'F390W': 25.1413, 'F438W': 24.9738,
                      'F475W': 25.7783, 'F555W': 25.8160, 'F606W': 25.9866,
                      'F625W': 25.3783, 'F775W': 24.4747, 'F814W': 24.6803,
                      'F390M': 23.5377, 'F410M': 23.7531, 'FQ422M': 22.9611,
                      'F467M': 23.8362, 'F547M': 24.7477, 'F621M': 24.4539,
                      'F689M': 24.1873, 'F763M': 23.8283, 'F845M': 23.2809}


def getzpt(image, system='Vega', ext=0):
    """ Define the zero point for the given image in the photometric system
    specified ("Vega", 'AB', "STMAG").

    :param image: any valid input to getheader(), namely:
      a string giving a fits filename, a pyfits hdulist or hdu, a pyfits
      header object, a tuple or list giving [hdr,data]
    :param system :  photometry system to use ('AB','Vega')

    :return: float zeropoint magnitude
    """
    import numpy as np

    header = getheader(image, ext=ext)
    filt = getfilter(header)
    camera = getcamera(header)
    if camera == 'ACS-WFC':
        # For ACS the zero point varies with time, so we interpolate
        # from a fixed table for either AB or Vega.
        ZEROPOINT = getzptACS(header, system=system)
    elif system.lower().startswith('ab'):
        # For AB mags, assuming the data are retreived from the archive
        # after 2012, we can use the header keywords to get the best zeropoint
        if ('PHOTPLAM' in header) and ('PHOTFLAM' in header):
            PHOTPLAM = header['PHOTPLAM']
            PHOTFLAM = header['PHOTFLAM']
        ZEROPOINT = -2.5 * np.log10(PHOTFLAM) - 5 * np.log10(PHOTPLAM) - 2.408
    elif system.lower().startswith('vega'):
        # For WFC3, the Vega zeropoint is read from a fixed table
        if camera == 'WFC3-UVIS':
            ZEROPOINT = ZPT_WFC3_UVIS_VEGA[filt]
        elif camera == 'WFC3-IR':
            ZEROPOINT = ZPT_WFC3_IR_VEGA[filt]
    return ZEROPOINT


def getzptACS(image, system='Vega', ext=0):
    """ Determine the ACS zeropoint for the given image by interpolating over
    a table of zeropoints. System may be 'Vega', 'ST', or 'AB'.

    :param image: any valid input to getheader(), namely:
      a string giving a fits filename, a pyfits hdulist or hdu, a pyfits
      header object, a tuple or list giving [hdr,data]
    :param system :  photometry system to use ('AB','Vega','STMAG')

    :return: float zeropoint magnitude

    """
    import os
    import numpy as np
    from astropy.io import ascii
    from scipy import interpolate as scint

    hdr = getheader(image, ext=ext)
    filtim = getfilter(hdr)
    mjdim = hdr['EXPSTART']

    thisfile = sys.argv[0]
    if 'ipython' in thisfile:
        thisfile = __file__
    thisdir = os.path.dirname(thisfile)
    acszptdatfile = os.path.join(thisdir, 'acs_wfc_zpt.dat')
    zptdat = ascii.read(acszptdatfile, format='commented_header',
                        header_start=-1, data_start=0)
    ifilt = np.where(zptdat['FILTER'] == filtim)
    if system.lower().startswith('vega'):
        zpt = zptdat['VEGAMAG'][ifilt]
    elif system.lower().startswith('st'):
        zpt = zptdat['STMAG'][ifilt]
    elif system.lower().startswith('ab'):
        zpt = zptdat['ABMAG'][ifilt]
    else:
        raise exceptions.RuntimeError(
            "Magnitude system %s not recognized" % system)
    mjd = zptdat['MJD'][ifilt]
    zptinterp = scint.interp1d(mjd, zpt, bounds_error=True)
    zptim = zptinterp(mjdim)

    return zptim


def dophot(image, xc, yc, aparcsec=0.4, system='AB', ext=None,
           psfimage=None, psfradpix=3, recenter=False, imfilename=None,
           ntestpositions=100, snthresh=0.0, zeropoint=None,
           filtername=None, exptime=None,
           skyannarcsec=[6.0, 12.0], skyval=None, skyalgorithm='sigmaclipping',
           target=None, printstyle=None, exact=False, fitsconvention=True,
           phpadu=None, returnflux=False, verbose=False, debug=False):
    """ Measure the flux through aperture(s) and/or psf fitting and report
    observed fluxes and magnitudes.

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
    import numpy as np
    import hstapcorr

    if debug == 1:
        import pdb
        pdb.set_trace()

    imhdr, imdat = get_header_and_data(image, ext=ext)
    if imfilename is None:
        if isinstance(image, basestring):
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

    pixscale = getpixscale(imhdr, ext=ext)
    if not np.iterable(aparcsec):
        aparcsec = np.array([aparcsec])
    elif not isinstance(aparcsec, np.ndarray):
        aparcsec = np.array(aparcsec)

    appix = np.array([ap / pixscale for ap in aparcsec])
    skyannpix = np.array([skyrad / pixscale for skyrad in skyannarcsec])
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

    if zeropoint is not None:
        zpt = zeropoint
        apcor = np.zeros(len(aparcsec))
        aperr = np.zeros(len(aparcsec))
    else:
        zpt = getzpt(image, system=system)
        if camera == 'WFC3-IR':
            apcor, aperr = hstapcorr.apcorrWFC3IR(filtername, aparcsec)
        elif camera == 'WFC3-UVIS':
            apcor, aperr = hstapcorr.apcorrWFC3UVIS(filtername, aparcsec)
        elif camera == 'ACS-WFC':
            apcor, aperr = hstapcorr.apcorrACSWFC(filtername, aparcsec)

    if fitsconvention:
        xpy, ypy = xc - 1, yc - 1
    else:
        xpy, ypy = xc, yc

    photoutput = photfunctions.get_flux_and_err(
        imdat, psfimage, [xpy, ypy],
        psfradpix=psfradpix, apradpix=appix, ntestpositions=ntestpositions,
        skyannpix=skyannpix, skyalgorithm=skyalgorithm, setskyval=skyval,
        recenter_target=recenter, recenter_fakes=True, exact=exact,
        exptime=exptime, ronoise=1, phpadu=phpadu, verbose=verbose,
        debug=debug)
    apflux, apfluxerr, psfflux, psffluxerr, sky, skyerr = photoutput
    if not np.iterable(apflux):
        apflux = np.array([apflux])
        apfluxerr = np.array([apfluxerr])

    if psfimage is not None:
        # record the psf flux as a final infinite aperture for printing
        # purposes:
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
                print " FERRTOT  FERRSTAT   FERRSYS"
                print " %.5f  %.5f  %.5f" % (apfluxerr[i], df, dfap)

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
        print 'VARLIST:  MJD  FLT FIELD   FLUXCAL   FLUXCALERR    MAG     '\
              'MAGERR   ZPT'
    elif verbose:
        if printstyle.lower() in ['long', 'verbose']:
            print '#  TARGET                RA         DEC       MJD  FILTER '\
                  ' APER       FLUX  FLUXERR         MAG   MAGERR  MAGSYS    '\
                  '   ZP      SKY SKYERR  IMAGE'
        else:
            print '# MJD     FILTER  APER      FLUX   FLUXERR       MAG     '\
                  'MAGERR  MAGSYS    ZP       SKY   SKYERR'

    if printstyle is not None :
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
    from numpy import array
    from astropy.io import fits as pyfits

    parser = argparse.ArgumentParser(
        description='Measure aperture photometry on drizzled HST images.'
                    ' using the PyIDLPhot routines.')

    # Required positional argument
    parser.add_argument('image', help='Drizzled HST image fits file.')
    parser.add_argument('x', type=float, help='X position or R.A.')
    parser.add_argument('y', type=float, help='Y position or Dec')
    parser.add_argument('--psfmodel', type=str, default=None,
                        help="Filename of a psf model fits file.")
    parser.add_argument('--ntest', type=int, default=None,
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
    parser.add_argument('--exact', action='store_true',
                        help='Compute sub-pixel areas exactly (Slower. '
                             'Necessary for small apertures.)')
    parser.add_argument('--apertures', type=str, default='0.4',
                        help='Size of photometry aperture(s) in arcsec. ')
    parser.add_argument('--filtername', type=str, default=None,
                        help='HST filter name (if not provided, we will read '
                             'it from the header)')
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
    parser.add_argument('-v', dest='verbose', action='count',
                        help='Turn verbosity up (use -v,-vv,-vvv, etc.)')
    parser.add_argument('-d', dest='debug', action='count',
                        help='Turn up debugging depth (use -d,-dd,-ddd)')

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
        if argv.verbose > 1:
            print("x,y position [fits-style, (1,1)-indexed]: %.2f %.2f" % (
                xim, yim))
    else:
        xim, yim = argv.x, argv.y

    if not argv.forced:
        xim, yim = getxycenter(argv.image, xim, yim, ext=argv.ext,
                               fitsconvention=True, radec=False,
                               verbose=argv.verbose)
        if argv.verbose:
            print("Recentered position (x,y) : %.2f %.2f" % (xim, yim))
            ra, dec = xy2radec(argv.image, xim, yim, ext=argv.ext)
            print("Recentered position (ra,dec) : %.6f %.6f" % (ra, dec))

    magsys = 'AB'
    if argv.vega:
        magsys = 'Vega'
    if argv.AB:
        magsys = 'AB'

    aplist = array([float(ap) for ap in argv.apertures.split(',')])
    skyannarcsec = [float(ap) for ap in argv.skyannulus.split(',')]
    maglinelist = dophot(argv.image, xim, yim, aplist,
                         psfimage=argv.psfmodel, ext=argv.ext,
                         skyannarcsec=skyannarcsec, skyval=argv.skyval,
                         system=magsys, zeropoint=argv.zeropoint,
                         filtername=argv.filtername,
                         skyalgorithm=argv.skyalgorithm,
                         snthresh=argv.snthresh,
                         exact=argv.exact,
                         ntestpositions=argv.ntest,
                         recenter=False,  # recentering already done above
                         printstyle=argv.printstyle, target=argv.target,
                         phpadu=argv.phpadu, verbose=argv.verbose,
                         debug=argv.debug)
    for iap in range(len(maglinelist)):
        print maglinelist[iap].strip()


if __name__ == '__main__':
    main()
