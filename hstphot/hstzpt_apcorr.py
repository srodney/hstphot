#! /usr/bin/env python
# 2014.07.01
# S.Rodney
# Determine zero points and compute aperture corrections from Encircled
# Energy tables or P330E photometry
__author__ = 'srodney'

import hstphot
import os
import sys
import numpy as np


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
    header = hstphot.getheader(image, ext=ext)
    filt = hstphot.getfilter(header)
    camera = hstphot.getcamera(header)
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
    from astropy.io import ascii
    from scipy import interpolate as scint

    hdr = hstphot.getheader(image, ext=ext)
    filtim = hstphot.getfilter(hdr)
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


def apcorrACSWFC( filter, aprad_arcsec, eetable='default', verbose=True ):
    """  Return the aperture correction for an ACS-WFC filter, derived from
    the STScI encircled energy tables.  If the aperture is 0.5" or 1.0" then
    the returned aperture correction is straight from the tables. Otherwise,
    we define the encircled energy fraction using linear interpolation
    between 5 fixed radii : (0.0, 1sigma, 0.5", 1.0", 5.5"), and then compute
    the approximate aperture correction from there.
    We adopt a systematic uncertainty of 1% for the 0.5" and 1.0" apertures,
    and increase this to 4% for apertures far from the interpolation anchor
    points.
    """
    from scipy import interpolate as scint

    #TODO: Allow user to input an encircled energy table as an external file

    # Encircled energy fractions for ACS-WFC filters in 0.5" and 1.0" apertures
    # from Bohlin's ISR-II (2011) and ISR-IV (2012).
    # retrieved from http://www.stsci.edu/hst/acs/analysis/apcorr  on 2014.07.04
    EE_ACS_WFC_05 = {'F435W':0.909,'F475W':0.910,'F502N':0.911,
                     'F555W':0.913,'F550M':0.914,'F606W':0.916,
                     'F625W':0.918,'F658N':0.919,'F660N':0.919,
                     'F775W':0.918,'F814W':0.914,'F892N':0.899,
                     'F850LP':0.893 }
    EE_ACS_WFC_10 = {'F435W':0.942,'F475W':0.943,'F502N':0.944,
                      'F555W':0.945,'F550M':0.946,'F606W':0.947,
                      'F625W':0.948,'F658N':0.949,'F660N':0.950,
                      'F775W':0.950,'F814W':0.948,'F892N':0.942,
                      'F850LP':0.939 }

    acsFWHM  = 0.14
    psfSigma = acsFWHM / 2.3548
    EEfrac00 = 0.0
    EEfrac01 = 0.68
    EEfrac05 = EE_ACS_WFC_05[filter]
    EEfrac10 = EE_ACS_WFC_10[filter]
    EEfrac55 = 1.0

    EEfracInterp = scint.interp1d(
        [0,psfSigma,0.5,1.0,5.5],
        [EEfrac00,EEfrac01,EEfrac05,EEfrac10,EEfrac55],
        kind='linear' )
    aperrInterp = scint.interp1d(
        [0,psfSigma,np.mean([psfSigma,0.5]),0.5,0.75, 1.0, 5.5],
        [0.04,0.025,             0.04,     0.01,0.04,0.01,0.01],
        kind='linear')

    EEfrac = EEfracInterp( aprad_arcsec )
    aperr  = aperrInterp( aprad_arcsec )
    apcor = -2.5*np.log10( EEfrac )

    # TODO : print a warning if the aperture is far from 0.5 or 1.0
    if verbose>1 :
        from matplotlib import pyplot as pl
        apsize = np.arange(0.0,5.5,0.01)
        pl.clf()
        apcorline = -2.5*np.log10(EEfracInterp(apsize))
        aperrline = aperrInterp( apsize )
        apcorpts = -2.5*np.log10([EEfrac00,EEfrac01,EEfrac05,EEfrac10,EEfrac55])
        pl.plot( apsize, apcorline, 'k-', label='filter' )
        pl.plot( [0.0,psfSigma,0.5,1.0,5.5], apcorpts, 'ro',ls=' ')
        pl.fill_between( apsize, apcorline+aperrline, apcorline-aperrline,
        color='c', alpha=0.3 )
        pl.draw()

    return( apcor, aperr )


def apcorrWFC3UVIS( filt, aprad_arcsec ) :
    """ compute the aperture correction and associated
    error for the given filter and camera, at the given
    aperture radius (in arcseconds), derived from the STScI
    encircled energy tables.
    """
    #  To compute aperture corrections from encircled energy::
    #   fobs = ftrue * EEfrac
    #   mobs = -2.5*log10( fobs ) + ZPT
    #   mtrue = -2.5*log10( ftrue ) + ZPT
    #
    #   apcor = mobs - mtrue
    #         = -2.5 * (  log10( fobs ) - log10( ftrue ) )
    #         = -2.5 * log10( fobs / ftrue )
    #   apcor = -2.5 * log10( EEfrac )

    from scipy import interpolate as scint
    import numpy as np
    from string import ascii_letters, punctuation

    # central filter wavelength, in um, for WFC3-UVIS filter names
    if filt.lower() == 'f350lp' : filt='600' # fudge for long-pass F350LP
    filtwave = int(filt.strip( ascii_letters + punctuation ))  / 1000.
    if np.iterable(aprad_arcsec) and not np.iterable(filtwave):
        filtwave = [filtwave for i in range(len(aprad_arcsec))]

    # wavelengths and aperture sizes (in arcsec) for the x and y
    # dimensions of the WFC3-UVIS encircled energy table, respectively
    wl = [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    ap = [ 0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0, 1.5, 2.0]

    # The encircled energy table, from
    # http://www.stsci.edu/hst/wfc3/phot_zp_lbn
    ee = np.array( [
    [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
    [ 0.660, 0.739, 0.754, 0.745, 0.720, 0.687, 0.650, 0.623, 0.612, 0.605 ],
    [ 0.717, 0.793, 0.823, 0.834, 0.832, 0.823, 0.807, 0.778, 0.742, 0.699 ],
    [ 0.752, 0.822, 0.845, 0.859, 0.859, 0.857, 0.853, 0.847, 0.844, 0.829 ],
    [ 0.781, 0.844, 0.864, 0.875, 0.877, 0.874, 0.870, 0.867, 0.868, 0.864 ],
    [ 0.802, 0.858, 0.880, 0.888, 0.890, 0.889, 0.883, 0.879, 0.879, 0.876 ],
    [ 0.831, 0.880, 0.899, 0.911, 0.910, 0.907, 0.906, 0.904, 0.900, 0.894 ],
    [ 0.861, 0.894, 0.912, 0.923, 0.925, 0.923, 0.918, 0.915, 0.918, 0.917 ],
    [ 0.884, 0.906, 0.922, 0.932, 0.934, 0.933, 0.931, 0.927, 0.927, 0.923 ],
    [ 0.936, 0.928, 0.936, 0.944, 0.947, 0.946, 0.945, 0.942, 0.944, 0.942 ],
    [ 0.967, 0.946, 0.948, 0.954, 0.955, 0.955, 0.955, 0.952, 0.955, 0.952 ],
    [ 0.989, 0.984, 0.973, 0.970, 0.970, 0.969, 0.967, 0.966, 0.970, 0.968 ],
    [ 0.994, 0.992, 0.989, 0.985, 0.980, 0.977, 0.976, 0.975, 0.978, 0.976 ], ] )

    ee_interp = scint.interp2d( wl, ap, ee, kind='linear', bounds_error=True )
    EEfrac_ap = ee_interp( filtwave, aprad_arcsec  ).reshape( np.shape(aprad_arcsec) )
    apcor = -2.5 * np.log10( EEfrac_ap ) # correction relative to infinite aperture
    if np.iterable( apcor ) :
        aperr = 0.005*np.ones(len(apcor))
    else :
        aperr = 0.005
    return( np.round(apcor,3), aperr )


def apcorrWFC3IR( filt, aprad_arcsec, eetable='default'):
    """ compute the aperture correction and associated
    error for the given WFC3-IR filter, at the given
    aperture size (in arcseconds), derived from the STScI
    encircled energy tables.

    :param filt: Name of the WFC3IR filter (e.g., 'F105W')
    :param aprad_arcsec: aperture radius in arcsec
    :param eetable: 'default', 'stsci', or filename;
       the Encircled Energy table to use.
       'default' is a table derived from broadband HST observations of the
       standard star P330E.
       'stsci' is the table posted on the stsci website.
       A filename can be provided, giving the EE table as an ascii text file.
       (see the function read_eetable() for formatting details)
    """
    from scipy import interpolate as scint
    import numpy as np
    from string import ascii_letters, punctuation

    # central filter wavelength, um, for WFC3-IR filter names
    filtwave = int(filt.strip( ascii_letters+punctuation ))  / 100.
    if not np.iterable(aprad_arcsec):
        aprad_arcsec = np.array([aprad_arcsec,])
    if not np.iterable(filtwave):
        filtwave = np.array([filtwave for i in range(len(aprad_arcsec))])

    if eetable == 'p330e':
        # Derived from measurements of drizzled HST images in broadband
        # WFC3-IR filters of the standard star P330E.
        # wavelengths and aperture sizes (in arcsec) for the x and y
        # dimensions of the WFC3-IR encircled energy table, respectively
        wl = [1.05, 1.10, 1.25, 1.40, 1.60]
        ap = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
              0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.15, 1.2, 1.25,
              1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85,
              1.9, 1.95, 2.0, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.6,
              2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0, ]

        # Note: factor of 0.978 rescales to match wide-aperture limit from
        # published STScI EE tables.
        ee = 0.978 * np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.403728, 0.4023152, 0.392663, 0.3677491, 0.346245],
            [0.618763, 0.6039498, 0.573211, 0.54726, 0.5196374],
            [0.739931, 0.7250895, 0.701141, 0.6687310, 0.6314328],
            [0.812651, 0.805773, 0.789060, 0.7635499, 0.7260767],
            [0.841463, 0.84245, 0.834553, 0.8234469, 0.8018365],
            [0.85861, 0.860857, 0.854606, 0.8514960, 0.8390703],
            [0.873424, 0.8740914, 0.86682, 0.8651446, 0.8560004],
            [0.887876, 0.8870479, 0.877837, 0.8747432, 0.8656558],
            [0.901487, 0.9000408, 0.890035, 0.8840476, 0.8731908],
            [0.911629, 0.9113724, 0.902988, 0.8948841, 0.8807754],
            [0.918863, 0.9202818, 0.914028, 0.9062657, 0.8901587],
            [0.925090, 0.927873, 0.922084, 0.9166304, 0.9014134],
            [0.930917, 0.9336749, 0.927612, 0.9244361, 0.9121914],
            [0.93695, 0.9392306, 0.93233, 0.9301586, 0.920455],
            [0.942678, 0.943964, 0.937026, 0.9346465, 0.9260929],
            [0.947757, 0.9486786, 0.941694, 0.9388042, 0.9302250],
            [0.952290, 0.9532832, 0.946412, 0.9431365, 0.9336466],
            [0.956459, 0.9576440, 0.950873, 0.9473226, 0.9371014],
            [0.960229, 0.9614003, 0.954860, 0.9512798, 0.9409858],
            [0.966726, 0.9683840, 0.961593, 0.9588100, 0.9497151],
            [0.969669, 0.9710078, 0.964730, 0.9620230, 0.9538715],
            [0.972835, 0.9736977, 0.967764, 0.9650033, 0.9574028],
            [0.975414, 0.976091, 0.970745, 0.9680107, 0.9604832],
            [0.977860, 0.9792678, 0.973515, 0.9707490, 0.9632085],
            [0.979931, 0.9817075, 0.976094, 0.9733250, 0.9656435],
            [0.981957, 0.9836032, 0.978319, 0.9757455, 0.9679088],
            [0.983823, 0.9855169, 0.980397, 0.9781824, 0.9702889],
            [0.98558, 0.9868250, 0.982223, 0.9805524, 0.9727660],
            [0.987166, 0.9882622, 0.98392, 0.9827457, 0.9751296],
            [0.988371, 0.9895518, 0.985576, 0.9848194, 0.9774166],
            [0.989258, 0.9912810, 0.986978, 0.9864400, 0.979441],
            [0.990026, 0.9923256, 0.988210, 0.9878730, 0.9812408],
            [0.991049, 0.9930831, 0.989520, 0.9891984, 0.9829366],
            [0.992267, 0.994084, 0.990733, 0.99035, 0.9843638],
            [0.993172, 0.9951880, 0.991887, 0.9914765, 0.9856712],
            [0.993933, 0.9960312, 0.992797, 0.9924993, 0.9871083],
            [0.99452, 0.9967595, 0.993535, 0.9933365, 0.9884438],
            [0.995008, 0.9973850, 0.994303, 0.994189, 0.9895333],
            [0.996270, 0.9972981, 0.99548, 0.9957798, 0.9916342],
            [0.996344, 0.9973483, 0.995920, 0.9965069, 0.9927223],
            [0.996738, 0.9980915, 0.996288, 0.9969584, 0.9937033],
            [0.996961, 0.9984655, 0.996719, 0.9975789, 0.9943866],
            [0.997819, 0.997497, 0.997126, 0.9980887, 0.9950500],
            [0.998544, 0.9973113, 0.997408, 0.9984955, 0.9958421],
            [0.998927, 0.997642, 0.997686, 0.998850, 0.9966254],
            [0.998826, 0.9975910, 0.997981, 0.9991443, 0.9972190],
            [0.999157, 0.9970780, 0.998701, 0.9994660, 0.9981056],
            [0.999163, 0.9961149, 0.998849, 0.9995665, 0.9984596],
            [0.999437, 0.9957332, 0.999014, 0.999685, 0.9986348],
            [0.999644, 0.9967060, 0.99909, 0.9997373, 0.9989475],
            [1.00004, 0.9970499, 0.999297, 0.9997164, 0.9991965],
            [1.00040, 0.9974887, 0.999651, 0.9997865, 0.9994882],
            [1.00032, 0.998499, 0.999766, 0.9999081, 0.999852],
            [1.0003, 0.9994538, 0.999893, 0.9998680, 0.9999760],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])

    elif eetable in ['default', 'stsci']:
        # wavelengths and aperture sizes (in arcsec) for the x and y
        # dimensions of the WFC3-IR encircled energy table, respectively
        wl = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
        ap = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80,
              1.0, 1.5, 2.0, 5.5]

        # The encircled energy table, from
        # http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir07.html#401707
        # http://www.stsci.edu/hst/wfc3/phot_zp_lbn
        ee = np.array([
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
             0.000, 0.000],
            [0.575, 0.549, 0.524, 0.502, 0.484, 0.468, 0.453, 0.438, 0.426,
             0.410, 0.394],
            [0.736, 0.714, 0.685, 0.653, 0.623, 0.596, 0.575, 0.558, 0.550,
             0.539, 0.531],
            [0.802, 0.794, 0.780, 0.762, 0.739, 0.712, 0.683, 0.653, 0.631,
             0.608, 0.590],
            [0.831, 0.827, 0.821, 0.813, 0.804, 0.792, 0.776, 0.756, 0.735,
             0.708, 0.679],
            [0.850, 0.845, 0.838, 0.833, 0.828, 0.822, 0.816, 0.808, 0.803,
             0.789, 0.770],
            [0.878, 0.876, 0.869, 0.859, 0.850, 0.845, 0.841, 0.838, 0.840,
             0.836, 0.832],
            [0.899, 0.894, 0.889, 0.884, 0.878, 0.868, 0.858, 0.852, 0.854,
             0.850, 0.848],
            [0.916, 0.913, 0.904, 0.897, 0.893, 0.889, 0.883, 0.875, 0.870,
             0.863, 0.859],
            [0.937, 0.936, 0.929, 0.924, 0.918, 0.909, 0.903, 0.900, 0.903,
             0.900, 0.895],
            [0.951, 0.951, 0.946, 0.941, 0.935, 0.930, 0.925, 0.920, 0.917,
             0.912, 0.909],
            [0.967, 0.969, 0.967, 0.965, 0.963, 0.959, 0.954, 0.951, 0.952,
             0.948, 0.943],
            [0.974, 0.977, 0.976, 0.975, 0.973, 0.972, 0.969, 0.967, 0.970,
             0.967, 0.963],
            [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
             1.000, 1.000],
        ])
    else:
        wl, ap, ee = read_eetable(eetable)

    ee_interp = scint.interp2d(wl, ap, ee, kind='cubic',
                               bounds_error=False, fill_value=1.0)
    EEfrac_ap = ee_interp(filtwave, aprad_arcsec)
    if len(EEfrac_ap.shape)>1:
        EEfrac_ap = EEfrac_ap[:,0]

    # magnitude to add to measured mag, to correct to an infinite aperture
    apcor = -2.5 * np.log10(EEfrac_ap)

    if np.iterable(apcor):
        aperr = 0.005*np.ones(len(apcor))
    else:
        aperr = 0.005
    return np.round(apcor, 3), aperr


# The following has some data and aperture correction definitions derived
# from P330E data for WFC3-IR.

thisfile = sys.argv[0]
if 'ipython' in thisfile :
    thisfile = __file__
thisdir = os.path.dirname( thisfile )
p330edatfile = os.path.join(thisdir,'p330e.dat')

apcordatfile = os.path.join(thisdir,'apcorrP330E.dat')

P330E_SynMagAB = {'F098M':12.661,'F105W':12.677,'F110W':12.694,'F125W':12.714,
                  'F127M':12.721,'F139M':12.738,'F140W':12.739,'F153M':12.763,
                  'F160W':12.776 }

def define_apcorr_p330e( p330edat ):
    """ Read in the p330e.dat file and define the aperture corrections
    """
    from matplotlib import pyplot as pl
    #from matplotlib import cm
    # from mpltools import color
    #from pytools import plotsetup

    filterlist = np.unique( p330edat['FILTER'] )
    aplist = np.unique( p330edat['APERTURE'])
    pixscalelist = np.unique( p330edat['PIXSCALE'])
    #color.cycle_cmap( len(filterlist), cmap=cm.gist_rainbow_r, ax=ax1)

    pl.clf()
    colorlist = ['darkorchid','lightblue','teal','lightgreen','darkgreen','darkorange','darkred','magenta','k']
    for filtname, color in zip(filterlist,colorlist) :

        isubset = np.array( [
            np.where( (p330edat['FILTER']==filtname) & (p330edat['PIXSCALE']==ps) )[0]
            for ps in pixscalelist ] )

        ms = p330edat['MSHORT'][isubset]
        mserr = p330edat['MSHORTERR'][isubset]
        mm = p330edat['MMED'][isubset]
        mmerr = p330edat['MMEDERR'][isubset]

        msdiff = ms - P330E_SynMagAB[filtname]
        mmdiff = mm - P330E_SynMagAB[filtname]

        apcorrs = msdiff.mean( axis=0 )
        apcorrserr = msdiff.std( axis=0 )

        apcorrm = mmdiff.mean( axis=0 )
        apcorrmerr = mmdiff.std( axis=0 )
        apsize = p330edat['APERTURE'][isubset].mean(axis=0)

        # to choose between the short and medium-length exposure sets,
        # we adopt the one that delivers the smallest aperture correction
        # in the largest aperture
        # and we define the uncertainty in the aperture correction as the
        # quadratic sum of the nominal aperture error (random)
        # and the value of the largest-aperture correction (systematic).
        ishortmed = np.argmin( [ apcorrs[-1], apcorrm[-1] ] )
        if abs(apcorrs[-1]) < abs(apcorrm[-1]):
            apcorr = apcorrs
            apcorrerr = np.sqrt( apcorrserr**2 + apcorrs[-1]**2 )
        else :
            apcorr = apcorrm
            apcorrerr = np.sqrt( apcorrmerr**2 + apcorrm[-1]**2 )

        # pl.errorbar( apsize[0], msdiff[0], mserr , marker='.', color='r', label='%s short'%filtname )
        # pl.errorbar( apsize[1], msdiff[1], mserr , marker='.', color='k', label='%s short'%filtname )
        pl.errorbar( apsize, apcorr, apcorrerr , marker='o', color=color, label=filtname )
        pl.draw()
        for iap in range(len(apsize)):
            print("%s  %.2f  %7.3f %7.3f"%(filtname, apsize[iap], apcorr[iap], apcorrerr[iap] ) )

def getapdat_P330E():
    from astropy.io import ascii
    p330edat = ascii.read( p330edatfile, format='commented_header', header_start=-1, data_start=0 )
    return( p330edat )

def apcorrWFC3IR_P330E(filter,aparcsec):
    """ Returns aperture corrections derived from the table of
    P330E aperture magnitudes.
    """
    from astropy.io import ascii
    from scipy import interpolate as scint
    apcordat = ascii.read( apcordatfile, format='commented_header', header_start=-1, data_start=0 )

    ifilt = np.where( apcordat['FILTER']==filter)
    apcorpts = apcordat['APCORR'][ ifilt ]
    errpts = apcordat['APCORRERR'][ ifilt ]
    apsize = apcordat['APERTURE'][ ifilt ]
    if apsize.max()<10.0 :
        apsize = np.append( apsize, 10.0 )
        apcorpts = np.append( apcorpts, 0.0 )
        errpts = np.append( errpts, errpts[-1] )
    if apsize.min()>0.001 :
        apcorslope = (apcorpts[1]-apcorpts[0])/(apsize[1]-apsize[0])
        apcor0 = apcorpts[0] - apcorslope*apsize[0]
        errslope = (errpts[1]-errpts[0])/(apsize[1]-apsize[0])
        err0 = errpts[0] - errslope*apsize[0]
        apsize = np.append( 0.001, apsize )
        errpts = np.append( err0, errpts )
        apcorpts = np.append( apcor0, apcorpts )

    apcorinterp = scint.interp1d( apsize, apcorpts, kind='linear', bounds_error=True )
    errinterp = scint.interp1d( apsize, errpts, kind='linear', bounds_error=True )

    return( apcorinterp( aparcsec ), errinterp( aparcsec ) )

def plotapcorr_P330E( filter='all' ):
    from matplotlib import pyplot as pl
    from astropy.io import ascii
    #from hstapphot import ZPT_WFC3_IR_AB_04
    #from hstapphot import ZPT_WFC3_IR_AB_INF

    apcordat =  ascii.read( apcordatfile, format='commented_header', header_start=-1, data_start=0 )
    apvals = np.arange(0.01, 5.0, 0.01 )
    if filter == 'all' : filtlist = np.unique( apcordat['FILTER'] )
    else : filtlist = np.array([ filter ])
    filtlist = np.sort( filtlist )
    pl.clf()
    naxis = len(filtlist)
    naxX = int( np.sqrt( naxis ) )
    naxY = naxis / naxX
    for iax in range(naxis) :
        filter = filtlist[iax]
        ax = pl.subplot( naxX, naxY, iax+1 )
        ifilt = np.where( apcordat['FILTER']==filter)
        #apcorSTSCI = ZPT_WFC3_IR_AB_INF[filter] - ZPT_WFC3_IR_AB_04[filter]
        #ax.plot( 0.4, apcorSTSCI, 'kD', ms=8 )

        apcorpts = apcordat['APCORR'][ ifilt ]
        errpts = apcordat['APCORRERR'][ ifilt ]
        apsize = apcordat['APERTURE'][ ifilt ]
        apcorline, errline = apcorrWFC3IR_P330E( filter, apvals )
        ax.plot( apvals, apcorline, color='m')
        ax.errorbar( apsize, apcorpts, errpts, color='m', marker='o', capsize=0, ls=' ',
                     label='measured from P330E')
        ax.fill_between( apvals, apcorline-errline, apcorline+errline, color='m', alpha=0.3 )

        ax.text(0.95,0.95,filter,ha='right',va='top',transform=ax.transAxes,fontsize='large')
        ax.set_xlim([0.12,0.72])
        ax.set_ylim([0.0,1.0])

        aparray = np.arange( 0.05, 1.5, 0.05 )
        apcor, apcorerr = apcorrWFC3IR(filter,aparray)
        ax.plot( aparray, apcor, color='k', ls='-.', lw=3 )

        if iax==0 :
            ax.text( 0.95, 0.45, 'Derived from STScI\n encircled energy curves', color='k', ha='right',va='bottom', transform=ax.transAxes )
            ax.text( 0.95, 0.25, 'Derived from P330E',ha='right',va='bottom', color='m', transform=ax.transAxes )
        #    ax.legend(numpoints=1,frameon=False)
    fig = pl.gcf()
    fig.suptitle( 'WFC3-IR Aperture Corrections')
    fig.text( 0.5, 0.05, 'aperture size (arcsec)', ha='center',va='center')
    fig.text( 0.05, 0.5, 'aperture correction (mag)', ha='center',va='center', rotation=90)


def read_eetable(filename):
    """ Read in an encircled energy table data file and report the results
    """
    from astropy.table import Table
    from string import ascii_letters, punctuation

    # TODO :  accommodate other data table formats
    eedat = Table.read(filename, format='ascii.fixed_width')
    ap = eedat['APER'].data

    # TODO : update to handle ACS filters too
    #  determine the wavelength in nm for each filter:
    filterlist = [colname.lstrip('E').lower() for colname in eedat.colnames
                  if not colname.lower().startswith('aper')]
    wl = [int(filtername.strip(ascii_letters + punctuation)) / 100.
          for filtername in filterlist]
    eedat.remove_column('APER')
    eedatarray = np.array([list(eerow) for eerow in eedat])

    return(wl, ap, eedatarray)


def main():
    import os
    import argparse
    from astropy.io import fits as pyfits

    parser = argparse.ArgumentParser(
        description=("Convert an Encircled Energy table into a list of "
                     " aperture corrections for specific HST filters."))


    # Required positional argument
    parser.add_argument('eetable', help='Encircled energy table data file.')
    parser.add_argument('--filters', type=str, default='F105W,F125W,F140W,F160W',
                        help="Comma-separated list of HST filters.")
    parser.add_argument('--apertures', type=str, default='0.1,0.2,0.3,0.4',
                        help='List of aperture(s) in arcsec. ')
    parser.add_argument('-v', dest='verbose', action='count', default=0,
                        help='Turn verbosity up (use -v,-vv,-vvv, etc.)')
    parser.add_argument('-d', dest='debug', action='count',
                        help='Turn up debugging depth (use -d,-dd,-ddd)')

    argv = parser.parse_args()

    filterlist = np.array([f.lower() for f in argv.filters.split(',')])
    aplist = np.array([float(ap) for ap in argv.apertures.split(',')])

    print('#filter  aperture  m_apcorr err_apcorr')
    for filtername in filterlist:
        apcorrlist = []
        for aparcsec in aplist:
            if filtername.lower().startswith('f1'):
                apcorr, apcorrerr = apcorrWFC3IR(filtername, aparcsec,
                                                 eetable = argv.eetable)
            else:
                #TODO : map filters to camera more carefully
                apcorr, apcorrerr = apcorrACSWFC(
                    filtername, aparcsec, eetable=argv.eetable)
            print('%7s  %7.2f   %7.3f  %7.3f' % (filtername, aparcsec, apcorr, apcorrerr))
            apcorrlist.append(apcorr)
        #print( '%s %s' % (filtername, str(apcorrlist)))


if __name__ == '__main__':
    main()
