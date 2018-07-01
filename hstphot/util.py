from astropy.io import fits as pyfits


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
