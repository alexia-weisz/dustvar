import os, sys
import astrogrid
import astropy.io.fits as pyfits
import numpy as np
import astropy.units as u
import astropy.constants as const
import scipy.ndimage as sp

from pdb import set_trace


FUV_LAMBDA = (1538.6 * u.angstrom).value  #from Morrissey et al. 2007 #1528?
NUV_LAMBDA = (2315.7 * u.angstrom).value  #2271
MIPS24_LAMBDA = (24 * u.micron).to(u.angstrom).value
IRAC1_LAMBDA = (3.6 * u.micron).to(u.angstrom).value
C = const.c.to('angstrom/s').value
MJYSR2JYARCSEC = 2.350443e-5
ALPHA_FUV = 8.0e-4
ALPHA_24MICRON = 0.1

def define_dir_structure(res):
    if os.environ['PATH'][1:6] == 'astro':
        _TOP_DIR = '/astro/store/phat/arlewis/'
    else:
        _TOP_DIR = '/Users/alexialewis/research/PHAT/'

    res_dir = 'res_' + res + 'pc/'

    _MOD_DIR = os.path.join(_TOP_DIR, 'maps', 'analysis')
    _DATA_DIR = os.path.join(_TOP_DIR, 'maps', 'analysis', res_dir)
    _WORK_DIR = os.path.join(_TOP_DIR, 'uvflux')

    return _DATA_DIR, _WORK_DIR, _MOD_DIR


def gather_map_data(res='90', dust_curve='cardelli', sfh='full_sfh'):

    _DATA_DIR, _WORK_DIR, _MOD_DIR = define_dir_structure(res)

    obsfiles = ['galex_fuv_test.fits', 'galex_fuv0.fits',
                'galex_fuv_nobgsub.fits', 'galex_nuv_test.fits',
                'galex_nuv0.fits','galex_nuv_nobgsub.fits']

    modfiles = ['mod_fuv_red.fits', 'mod_fuv_int.fits',
                'mod_nuv_red.fits', 'mod_nuv_int.fits',
                'mod_fuv_red_uncs_lower.fits', 'mod_fuv_red_uncs_upper.fits',
                'mod_fuv_int_uncs_lower.fits', 'mod_fuv_int_uncs_upper.fits',
                'mod_nuv_red_uncs_lower.fits', 'mod_nuv_red_uncs_upper.fits',
                'mod_nuv_int_uncs_lower.fits', 'mod_nuv_int_uncs_upper.fits']

    otherfiles = ['sfr100.fits', 'dust.fits', 'dust_av.fits','dust_dav.fits',
                  'irac1_mass.fits', 'mips_24_MJysr.fits']

    uvdustfiles = ['A_nuv.fits', 'A_fuv.fits']


    weightfile = os.path.join(_DATA_DIR, 'weights_orig.fits')
    w, h = pyfits.getdata(weightfile, header=True)
    w[(w > 0) & (w < 0.95)] = 0
    sel = (w == 0)

    fuvfluxobs, fuvfluxobs_bg, fuvfluxobs_nobg = [pyfits.getdata(os.path.join(_DATA_DIR, x)) * w for x in obsfiles[0:3]]

    nuvfluxobs, nuvfluxobs_bg, nuvfluxobs_nobg = [pyfits.getdata(os.path.join(_DATA_DIR, x)) * w for x in obsfiles[3:6]]

    fuvfluxmodred, fuvfluxmodint = [pyfits.getdata(os.path.join(_DATA_DIR,
        dust_curve, sfh, x)) * w for x in modfiles[0:2]]

    nuvfluxmodred, nuvfluxmodint = [pyfits.getdata(os.path.join(_DATA_DIR,
        dust_curve, sfh, x)) * w for x in modfiles[2:4]]

    fuvfluxmodred_lower, fuvfluxmodred_upper = [pyfits.getdata(os.path.join(_DATA_DIR, dust_curve, sfh, x)) * w for x in modfiles[4:6]]

    fuvfluxmodint_lower, fuvfluxmodint_upper = [pyfits.getdata(os.path.join(_DATA_DIR, dust_curve, sfh, x)) * w for x in modfiles[6:8]]

    nuvfluxmodred_lower, nuvfluxmodred_upper = [pyfits.getdata(os.path.join(_DATA_DIR, dust_curve, sfh, x)) * w for x in modfiles[8:10]]

    nuvfluxmodint_lower, nuvfluxmodint_upper = [pyfits.getdata(os.path.join(_DATA_DIR, dust_curve, sfh, x)) * w for x in modfiles[10:12]]

    sfr100, avdav, av, dav, irac1mass, mips24 = [pyfits.getdata(os.path.join(_DATA_DIR, x)) * w for x in otherfiles]

    anuv, afuv = [pyfits.getdata(os.path.join(_WORK_DIR, x)) * w
                  for x in uvdustfiles]

    for data in [fuvfluxobs, nuvfluxobs, fuvfluxobs_bg, nuvfluxobs_bg,
                 fuvfluxobs_nobg, nuvfluxobs_nobg, fuvfluxmodint,
                 nuvfluxmodint, fuvfluxmodred, nuvfluxmodred,
                 fuvfluxmodred_lower, fuvfluxmodred_upper,
                 fuvfluxmodint_lower, fuvfluxmodint_upper,
                 nuvfluxmodred_lower, nuvfluxmodred_upper,
                 nuvfluxmodint_lower, nuvfluxmodint_upper,
                 sfr100, avdav, av, dav,  irac1mass, mips24, anuv, afuv]:
        data[sel] = np.nan

    sSFR = sfr100/irac1mass


    #convert 24 micron from MJy/sr to Jy/arcsec**2
    mips24 *= 2.350443e-5
    dx, dy = astrogrid.wcs.calc_pixscale(h, ref='crpix').arcsec
    pixel_area = dx * dy
    ## convert to Jy/pix
    mips24 *= pixel_area
    # mips24 map is in Jy --> convert to erg s-1 cm-2 A-1
    # nu f_nu = lambda f_lambda so f_lambda = c/lambda^2 f_nu

    mips24 *= 1e-23 #convert to erg s-1 cm-2 Hz-1
    c = const.c.to('angstrom/s').value
    l = (24 * u.micron).to(u.angstrom).value
    mips24 = c / l**2 * mips24


    fuvmagobs = astrogrid.flux.galex_flux2mag(fuvfluxobs, 'galex_fuv')
    fuvmagmodint = astrogrid.flux.galex_flux2mag(fuvfluxmodint, 'galex_fuv')
    fuvmagmodred = astrogrid.flux.galex_flux2mag(fuvfluxmodred, 'galex_fuv')
    nuvmagobs = astrogrid.flux.galex_flux2mag(nuvfluxobs, 'galex_nuv')
    nuvmagmodint = astrogrid.flux.galex_flux2mag(nuvfluxmodint, 'galex_nuv')
    nuvmagmodred = astrogrid.flux.galex_flux2mag(nuvfluxmodred, 'galex_nuv')


    fuvdata = {'fluxobs':fuvfluxobs, 'fluxobs_nobg':fuvfluxobs_nobg,
               'fluxobs_bg':fuvfluxobs_bg, 'fluxmodint':fuvfluxmodint,
               'fluxmodred':fuvfluxmodred,
               'fluxmodred_lower': fuvfluxmodred_lower,
               'fluxmodred_upper': fuvfluxmodred_upper,
               'fluxmodint_lower': fuvfluxmodint_lower,
               'fluxmodint_upper': fuvfluxmodint_upper,
               'magobs':fuvmagobs, 'magmodint':fuvmagmodint,
               'magmodred':fuvmagmodred, 'ext':afuv}
    nuvdata = {'fluxobs':nuvfluxobs, 'fluxobs_nobg':nuvfluxobs_nobg,
               'fluxobs_bg':nuvfluxobs_bg, 'fluxmodint':nuvfluxmodint,
               'fluxmodred':nuvfluxmodred,
               'fluxmodred_lower': nuvfluxmodred_lower,
               'fluxmodred_upper': nuvfluxmodred_upper,
               'fluxmodint_lower': nuvfluxmodint_lower,
               'fluxmodint_upper': nuvfluxmodint_upper,
               'magobs':nuvmagobs,
               'magmodint':nuvmagmodint, 'magmodred':nuvmagmodred, 'ext':anuv}

    otherdata = {'sfr100':sfr100, 'avdav':avdav, 'av':av, 'dav':dav,
                 'irac1mass':irac1mass, 'sSFR':sSFR, 'mips24':mips24}


    return fuvdata, nuvdata, otherdata


def gather_sfh(res, sfhcube='sfr_evo_cube.fits', metalcube=None):

    _DATA_DIR, _WORK_DIR, _MOD_DIR = define_dir_structure(res)

    sfhcubefile = _DATA_DIR + sfhcube
    sfhcubefile_upper = _DATA_DIR + 'sfr_evo_upper_cube.fits'
    sfhcubefile_lower = _DATA_DIR + 'sfr_evo_lower_cube.fits'
    sfhcube, sfhhdr = pyfits.getdata(sfhcubefile, header=True) #20 x 211 x 75
    sfhcube_upper = pyfits.getdata(sfhcubefile_upper)
    sfhcube_lower = pyfits.getdata(sfhcubefile_lower)

    if metalcube is not None:
        metalfile = _DATA_DIR + metalcube
        metalcube, metalhdr = pyfits.getdata(metalfile, header=True)
        return sfhcube, sfhcube_upper, sfhcube_lower, sfhhdr, metalcube

    return sfhcube, sfhcube_upper, sfhcube_lower, sfhhdr


def gather_map_data_agelim(res='90', dust_curve='cardelli', sfh='full_sfh', correct_obs=False):

    _DATA_DIR, _WORK_DIR, _MOD_DIR = define_dir_structure(res)

    obsfiles = ['galex_fuv_test.fits', 'galex_nuv_test.fits']

    fuvmodfiles = ['mod_fuv_red.fits', 'mod_fuv_int.fits',
                   'mod_fuv_red_8000.fits', 'mod_fuv_red_5000.fits',
                   'mod_fuv_red_4000.fits', 'mod_fuv_red_2000.fits',
                   'mod_fuv_red_1000.fits', 'mod_fuv_int_1000.fits',
                   'mod_fuv_red_750.fits', 'mod_fuv_int_750.fits',
                   'mod_fuv_red_500.fits', 'mod_fuv_int_500.fits',
                   'mod_fuv_red_400.fits', 'mod_fuv_int_400.fits',
                   'mod_fuv_red_300.fits', 'mod_fuv_int_300.fits',
                   'mod_fuv_red_200.fits', 'mod_fuv_int_200.fits',
                   'mod_fuv_red_100.fits', 'mod_fuv_int_100.fits']

    nuvmodfiles = ['mod_nuv_red.fits', 'mod_nuv_int.fits',
                   'mod_nuv_red_8000.fits', 'mod_nuv_red_5000.fits',
                   'mod_nuv_red_4000.fits', 'mod_nuv_red_2000.fits',
                   'mod_nuv_red_1000.fits', 'mod_nuv_int_1000.fits',
                   'mod_nuv_red_750.fits', 'mod_nuv_int_750.fits',
                   'mod_nuv_red_500.fits', 'mod_nuv_int_500.fits',
                   'mod_nuv_red_400.fits', 'mod_nuv_int_400.fits',
                   'mod_nuv_red_300.fits', 'mod_nuv_int_300.fits',
                   'mod_nuv_red_200.fits', 'mod_nuv_int_200.fits',
                   'mod_nuv_red_100.fits', 'mod_nuv_int_100.fits']

    otherfiles = ['sfr100.fits', 'mips_24_MJysr.fits', 'irac_1_MJysr.fits']


    weightfile = os.path.join(_DATA_DIR, 'weights_orig.fits')
    w, h = pyfits.getdata(weightfile, header=True)
    w[(w > 0) & (w < 0.95)] = 0
    sel = (w == 0)

    ## pixel size
    dx, dy = astrogrid.wcs.calc_pixscale(h, ref='crpix').arcsec
    pixel_area = dx * dy

    fuvfluxobs, nuvfluxobs = [pyfits.getdata(os.path.join(_DATA_DIR, x)) * w for x in obsfiles]

    fuvfluxmodred, fuvfluxmodint, fuvfluxmodred8000, fuvfluxmodred5000, fuvfluxmodred4000, fuvfluxmodred2000,  fuvfluxmodred1000, fuvfluxmodint1000, fuvfluxmodred750, fuvfluxmodint750, fuvfluxmodred500, fuvfluxmodint500, fuvfluxmodred400, fuvfluxmodint400, fuvfluxmodred300, fuvfluxmodint300, fuvfluxmodred200, fuvfluxmodint200, fuvfluxmodred100, fuvfluxmodint100 = [pyfits.getdata(os.path.join(_MOD_DIR, x)) * w for x in fuvmodfiles]

    nuvfluxmodred, nuvfluxmodint,nuvfluxmodred8000, nuvfluxmodred5000, nuvfluxmodred4000, nuvfluxmodred2000,  nuvfluxmodred1000, nuvfluxmodint1000, nuvfluxmodred750, nuvfluxmodint750, nuvfluxmodred500, nuvfluxmodint500, nuvfluxmodred400, nuvfluxmodint400, nuvfluxmodred300, nuvfluxmodint300, nuvfluxmodred200, nuvfluxmodint200, nuvfluxmodred100, nuvfluxmodint100 = [pyfits.getdata(os.path.join(_MOD_DIR, x)) * w for x in nuvmodfiles]

    sfr100, mips24, irac1 = [pyfits.getdata(os.path.join(_DATA_DIR, x)) * w for x in otherfiles]


    for data in [fuvfluxobs, nuvfluxobs, fuvfluxmodint, nuvfluxmodint,
                 fuvfluxmodred, nuvfluxmodred,
                 fuvfluxmodred8000, nuvfluxmodred8000,
                 fuvfluxmodred5000, nuvfluxmodred5000,
                 fuvfluxmodred4000, nuvfluxmodred4000,
                 fuvfluxmodred2000, nuvfluxmodred2000,
                 fuvfluxmodint1000, nuvfluxmodint1000,
                 fuvfluxmodred1000, nuvfluxmodred1000,
                 fuvfluxmodint750, nuvfluxmodint750, fuvfluxmodred750,
                 nuvfluxmodred750, fuvfluxmodint500, nuvfluxmodint500,
                 fuvfluxmodred500, nuvfluxmodred500, fuvfluxmodint400,
                 nuvfluxmodint400, fuvfluxmodred400, nuvfluxmodred400,
                 fuvfluxmodint300, nuvfluxmodint300, fuvfluxmodred300,
                 nuvfluxmodred300, fuvfluxmodint200, nuvfluxmodint200,
                 fuvfluxmodred200, nuvfluxmodred200, fuvfluxmodint100,
                 nuvfluxmodint100, fuvfluxmodred100, nuvfluxmodred100,
                 sfr100, mips24, irac1]:
        data[sel] = np.nan

    def local_mean(A):
        return np.nanmean(A)

    def filter(data):
        mean_filter = sp.filters.generic_filter(data, local_mean, size=3)
        return mean_filter

    if correct_obs:
        ## correct for foreground stars
        ##    NUV/FUV > 15 (MJy/sr) and NUV detection 5 sigma
        ## (1) Convert FUV and NUV to MJy/sr
        fuv_mjysr = fuvfluxobs / MJYSR2JYARCSEC / pixel_area / 1e-23 / C * FUV_LAMBDA**2
        nuv_mjysr = nuvfluxobs / MJYSR2JYARCSEC / pixel_area / 1e-23 / C * NUV_LAMBDA**2

        fuv_filter = filter(fuv_mjysr)
        nuv_filter = filter(nuv_mjysr)
        mips24_filter = filter(mips24)

        nuv_std = np.nanstd(nuv_mjysr)
        sig = 5 * nuv_std
        fg_sel = np.where((nuv_mjysr/fuv_mjysr > 15) & ((nuv_mjysr > np.nanmean(nuv_mjysr) + sig)) | (nuv_mjysr < np.nanmean(nuv_mjysr) - sig))

        fuv_mjysr[fg_sel] = fuv_filter[fg_sel]
        nuv_mjysr[fg_sel] = nuv_filter[fg_sel]
        mips24[fg_sel] = mips24_filter[fg_sel]

        ## correct for old stars
        ##   fuvdata (MJy/sr) -= ALPHA_FUV * 3.6 micron (MJy/sr)
        ##   24 micron (MJy/sr) -= ALPHA_24MICRON * 3.6 micron (MJy/sr)
        fuv_mjysr -= ALPHA_FUV * irac1
        mips24 -= ALPHA_24MICRON * irac1

        #convert 24 micron from MJy/sr to Jy/arcsec**2
        mips24 *= MJYSR2JYARCSEC
        newfuvobs = fuv_mjysr * MJYSR2JYARCSEC
        newnuvobs = nuv_mjysr * MJYSR2JYARCSEC

        ## convert to Jy/pix
        mips24 *= pixel_area
        newfuvobs *= pixel_area
        newnuvobs *= pixel_area

        # mips24 map is in Jy --> convert to erg s-1 cm-2 A-1
        # nu f_nu = lambda f_lambda so f_lambda = c/lambda^2 f_nu
        mips24 *= 1e-23 #convert to erg s-1 cm-2 Hz-1
        newfuvobs *= 1e-23
        newnuvobs *= 1e-23
        mips24 = C / MIPS24_LAMBDA**2 * mips24
        newfuvobs = C / FUV_LAMBDA**2 * newfuvobs
        newnuvobs = C / NUV_LAMBDA**2 * newnuvobs
    else:
        fuv_filter = filter(fuvfluxobs)
        nuv_filter = filter(nuvfluxobs)
        mips24_filter = filter(mips24)

        nuv_std = np.nanstd(nuvfluxobs)
        sig = 5 * nuv_std
        fg_sel = np.where((nuvfluxobs/fuvfluxobs > 15) & ((nuvfluxobs > np.nanmean(nuvfluxobs) + sig)) | (nuvfluxobs < np.nanmean(nuvfluxobs) - sig))

        fuvfluxobs[fg_sel] = fuv_filter[fg_sel]
        nuvfluxobs[fg_sel] = nuv_filter[fg_sel]
        mips24[fg_sel] = mips24_filter[fg_sel]

        mips24 *= MJYSR2JYARCSEC * pixel_area * 1e-23 * C / MIPS24_LAMBDA**2
        newfuvobs = fuvfluxobs
        newnuvobs = nuvfluxobs

    fuvmagobs = astrogrid.flux.galex_flux2mag(newfuvobs, 'galex_fuv')
    nuvmagobs = astrogrid.flux.galex_flux2mag(newnuvobs, 'galex_nuv')
    fuvmagmodint = astrogrid.flux.galex_flux2mag(fuvfluxmodint, 'galex_fuv')
    fuvmagmodred = astrogrid.flux.galex_flux2mag(fuvfluxmodred, 'galex_fuv')
    nuvmagmodint = astrogrid.flux.galex_flux2mag(nuvfluxmodint, 'galex_nuv')
    nuvmagmodred = astrogrid.flux.galex_flux2mag(nuvfluxmodred, 'galex_nuv')

    fuvdata = {'fluxobs':newfuvobs, 'magobs':fuvmagobs,
               'magmodred': fuvmagmodred, 'magmodint': fuvmagmodint,
               'fluxmodint':fuvfluxmodint, 'fluxmodred':fuvfluxmodred,
               'fluxmodred8000':fuvfluxmodred8000,
               'fluxmodred5000':fuvfluxmodred5000,
               'fluxmodred4000':fuvfluxmodred4000,
               'fluxmodred2000':fuvfluxmodred2000,
               'fluxmodint1000':fuvfluxmodint1000,
               'fluxmodred1000':fuvfluxmodred1000,
               'fluxmodint750':fuvfluxmodint750,
               'fluxmodred750':fuvfluxmodred750,
               'fluxmodint500':fuvfluxmodint500,
               'fluxmodred500':fuvfluxmodred500,
               'fluxmodint400':fuvfluxmodint400,
               'fluxmodred400':fuvfluxmodred400,
               'fluxmodint300':fuvfluxmodint300,
               'fluxmodred300':fuvfluxmodred300,
               'fluxmodint200':fuvfluxmodint200,
               'fluxmodred200':fuvfluxmodred200,
               'fluxmodint100':fuvfluxmodint100,
               'fluxmodred100':fuvfluxmodred100}
    nuvdata = {'fluxobs':newnuvobs, 'magobs': nuvmagobs,
               'magmodred': nuvmagmodred, 'magmodint': nuvmagmodint,
               'fluxmodint':nuvfluxmodint, 'fluxmodred':nuvfluxmodred,
               'fluxmodred8000':nuvfluxmodred8000,
               'fluxmodred5000':nuvfluxmodred5000,
               'fluxmodred4000':nuvfluxmodred4000,
               'fluxmodred2000':nuvfluxmodred2000,
               'fluxmodint1000':nuvfluxmodint1000,
               'fluxmodred1000':nuvfluxmodred1000,
               'fluxmodint750':nuvfluxmodint750,
               'fluxmodred750':nuvfluxmodred750,
               'fluxmodint500':nuvfluxmodint500,
               'fluxmodred500':nuvfluxmodred500,
               'fluxmodint400':nuvfluxmodint400,
               'fluxmodred400':nuvfluxmodred400,
               'fluxmodint300':nuvfluxmodint300,
               'fluxmodred300':nuvfluxmodred300,
               'fluxmodint200':nuvfluxmodint200,
               'fluxmodred200':nuvfluxmodred200,
               'fluxmodint100':nuvfluxmodint100,
               'fluxmodred100':nuvfluxmodred100}

    otherdata = {'sfr100':sfr100, 'mips24':mips24}

    return fuvdata, nuvdata, otherdata

