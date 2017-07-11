#! /usr/bin/env python
# 2014.07.01
# S.Rodney
# Compute aperture corrections from Encircled Energy tables or P330E photometry
__author__ = 'srodney'

import os
import sys
import numpy as np

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

def apcorrWFC3IR( filt, aprad_arcsec, eetable='default' ) :
    """ compute the aperture correction and associated
    error for the given WFC3-IR filter, at the given
    aperture size (in arcseconds), derived from the STScI
    encircled energy tables.
    """
    from scipy import interpolate as scint
    import numpy as np
    from string import ascii_letters, punctuation

    # central filter wavelength, um, for WFC3-IR filter names
    filtwave = int(filt.strip( ascii_letters+punctuation ))  / 100.


    if eetable == 'default':
        # wavelengths and aperture sizes (in arcsec) for the x and y
        # dimensions of the WFC3-IR encircled energy table, respectively
        wl = [ 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 ]
        ap = [ 0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0, 1.5, 2.0, 5.5 ]

        # The encircled energy table, from
        # http://www.stsci.edu/hst/wfc3/documents/handbooks/currentIHB/c07_ir07.html#401707
        # http://www.stsci.edu/hst/wfc3/phot_zp_lbn
        ee = np.array( [
        [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000 ],
        [ 0.575, 0.549, 0.524, 0.502, 0.484, 0.468, 0.453, 0.438, 0.426, 0.410, 0.394 ],
        [ 0.736, 0.714, 0.685, 0.653, 0.623, 0.596, 0.575, 0.558, 0.550, 0.539, 0.531 ],
        [ 0.802, 0.794, 0.780, 0.762, 0.739, 0.712, 0.683, 0.653, 0.631, 0.608, 0.590 ],
        [ 0.831, 0.827, 0.821, 0.813, 0.804, 0.792, 0.776, 0.756, 0.735, 0.708, 0.679 ],
        [ 0.850, 0.845, 0.838, 0.833, 0.828, 0.822, 0.816, 0.808, 0.803, 0.789, 0.770 ],
        [ 0.878, 0.876, 0.869, 0.859, 0.850, 0.845, 0.841, 0.838, 0.840, 0.836, 0.832 ],
        [ 0.899, 0.894, 0.889, 0.884, 0.878, 0.868, 0.858, 0.852, 0.854, 0.850, 0.848 ],
        [ 0.916, 0.913, 0.904, 0.897, 0.893, 0.889, 0.883, 0.875, 0.870, 0.863, 0.859 ],
        [ 0.937, 0.936, 0.929, 0.924, 0.918, 0.909, 0.903, 0.900, 0.903, 0.900, 0.895 ],
        [ 0.951, 0.951, 0.946, 0.941, 0.935, 0.930, 0.925, 0.920, 0.917, 0.912, 0.909 ],
        [ 0.967, 0.969, 0.967, 0.965, 0.963, 0.959, 0.954, 0.951, 0.952, 0.948, 0.943 ],
        [ 0.974, 0.977, 0.976, 0.975, 0.973, 0.972, 0.969, 0.967, 0.970, 0.967, 0.963 ],
        [ 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000 ],
        ] )
    else:
        wl, ap, ee = read_eetable(eetable)
    # import pdb; pdb.set_trace()

    ee_interp = scint.interp2d( wl, ap, ee, kind='linear', bounds_error=False, fill_value=1.0 )
    EEfrac_ap = ee_interp( filtwave, aprad_arcsec  ).reshape( np.shape(aprad_arcsec) )
    apcor = -2.5 * np.log10( EEfrac_ap ) # correction relative to infinite aperture

    if np.iterable( apcor ) :
        aperr = 0.005*np.ones(len(apcor))
    else :
        aperr = 0.005
    return( np.round(apcor,3), aperr )


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
    parser.add_argument('--AB', action='store_true',
                        help='Use AB mags (the default).')
    parser.add_argument('--vega', action='store_true',
                        help='Use Vega mags.')
    parser.add_argument('--apertures', type=str, default='0.1,0.2,0.3,0.4',
                        help='List of aperture(s) in arcsec. ')
    parser.add_argument('-v', dest='verbose', action='count', default=0,
                        help='Turn verbosity up (use -v,-vv,-vvv, etc.)')
    parser.add_argument('-d', dest='debug', action='count',
                        help='Turn up debugging depth (use -d,-dd,-ddd)')

    argv = parser.parse_args()


    magsys = 'AB'
    if argv.vega:
        magsys = 'Vega'
    if argv.AB:
        magsys = 'AB'

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
            print '%7s  %7.2f   %7.3f  %7.3f' % (filtername, aparcsec, apcorr, apcorrerr)
            apcorrlist.append(apcorr)
        #print( '%s %s' % (filtername, str(apcorrlist)))


if __name__ == '__main__':
    main()
