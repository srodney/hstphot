from astropy.io import fits as pyfits

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


def getheaderanddata(image, ext=None):
    """ Return a fits image header and data array.

    :param image: a string giving a fits filename, a pyfits hdulist or hdu,
      a pyfits primaryhdu object, a tuple or list giving [hdr,data]
    :param ext: (optional) extension number for science array
    :return:  pyfits.header.Header object and numpy data array
    """
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
        if len(image) == 2:
            hdr, data = image
        else:
            raise RuntimeError('Input list/tuple must have exactly'
                               ' 2 entries, giving [hdr,data]')
    else:
        raise RuntimeError('input object type %s is unrecognized')

    return hdr, data


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
            raise RuntimeError('input object type %s is unrecognized')
    return hdr


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


