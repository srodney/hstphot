#! /usr/bin/env python
# 2014.06.29  S.Rodney
__author__ = 'rodney'

from .util import *
from . import dophot

import sys
import numpy as np
if sys.version_info <= (3,0):
    import exceptions


def main():
    import os
    import argparse
    from numpy import array
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
    parser.add_argument('--weightim', default=None, type=str,
                        help='weight image')
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
        maglinelist = dophot.dopythonphot(
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
    elif argv.photpackage.lower() == 'photutils':
        targetim = dophot.doastropyphot(
            argv.image, [xim, yim], apradarcsec=aplist,
            psfimfilename=argv.psfmodel, zpt=argv.zeropoint)


if __name__ == '__main__':
    main()
