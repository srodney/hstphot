from os import path
import sys
import numpy as np
if sys.version_info <= (3,0):
    import exceptions

from . import util
from . import hstzpt_apcorr
from . import astropyphot
_HST_WFC3_PSF_FWHM_ARCSEC = 0.14  # FWHM of the HST WFC3IR PSF in arcsec


def doastropyphot(targetimfilename, coordinates, coord_type='sky',
                  psfimfilename=None, ext=0,
                  psfpixscale=None, recenter_target=True,
                  apradarcsec=[0.1,0.2,0.3], skyannradarcsec=[3.0,5.0],
                  fitpix=11, targetname='TARGET', zpt=None,
                  ntestpositions=100, psfradpix=3,
                  skyannpix=None, skyalgorithm='sigmaclipping',
                  setskyval=None, recenter_fakes=True,
                  exptime=1, exact=True, ronoise=1, phpadu=1, verbose=False,
                  debug=False):
    """ Measure the flux (and uncertainty?) using the astropy-affiliated
    photutils package.

    :param targetimfilename: name of the .fits image file with the target (the
      star to be photometered).
    :param coordinates: x,y or ra,dec coordinates
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
    if coord_type.lower() in ['sky', 'radec']:
        ra, dec = coordinates
        coordinates = util.radec2xy(targetimfilename, ra, dec, ext=ext)
    x, y = coordinates

    targetim = astropyphot.TargetImage(targetimfilename, zpt=zpt)
    if psfpixscale is None:
        psfpixscale = targetim.pixscale
    targetim.set_target(x_0=x, y_0=y, targetname=targetname,
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

    if verbose:
        photresults = targetim.phot_summary_table
        photresults.pprint()
    return targetim


def dopythonphot(image, xc, yc, aparcsec=0.4, system='AB', ext=None,
                 psfimage=None, psfradpix=3, recenter=False, imfilename=None,
                 ntestpositions=100, snthresh=0.0, zeropoint=None,
                 filtername=None, exptime=None, pixscale=None,
                 skyannarcsec=[6.0, 12.0], skyval=None,
                 skyalgorithm='sigmaclipping',
                 target=None, printstyle='default', exact=True, fitsconvention=True,
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
      printstyle :  'default' = report MJD, filter, and photometry
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

    printstyle = printstyle.lower()

    imhdr, imdat = util.getheaderanddata(image, ext=ext)
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
        pixscale = util.getpixscale(imhdr, ext=ext)
        if not np.iterable(aparcsec):
            aparcsec = np.array([aparcsec])
        elif not isinstance(aparcsec, np.ndarray):
            aparcsec = np.array(aparcsec)

    appix = np.array([ap / pixscale for ap in aparcsec])
    skyannpix = np.array([skyrad / pixscale for skyrad in skyannarcsec])
    if len(appix) >= 1:
        assert skyannpix[0] >= np.max(
            appix), "Sky annulus must be >= largest aperture."
    camera = util.getcamera(imhdr)

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
        xim, yim = util.getxycenter([imhdr, imdat], xc, yc,
                                        fitsconvention=True, radec=False,
                                        verbose=verbose)
        if verbose:
            print("Recentered position (x,y) : %.2f %.2f" % (xim, yim))
            ra, dec = util.xy2radec(imhdr, xim, yim)
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
        return apflux, apfluxerr

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

    ra, dec = 0, 0
    if (printstyle is not None and
                printstyle.lower() in ['snana', 'long', 'verbose']):
        if not target and 'FILENAME' in imhdr.keys():
            target = imhdr['FILENAME'].split('_')[0]
        elif not target:
            target = 'target'
        ra, dec = util.xy2radec(imhdr, xc, yc, ext=ext)

    maglinelist = []
    for iap in range(len(aparcsec)):
        if printstyle == 'snana':
            magline = 'OBS: %8.2f   %6s   %s %8.3f %8.3f    '\
                      '%8.3f %8.3f   %.3f' % (
                          float(mjdobs), util.FilterAlpha[filtername], target,
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
