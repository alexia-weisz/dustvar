import astrogrid.flux
import astropy.io.fits
import m31maps
import os


sfr100_map, hdr = astropy.io.fits.getdata(
    m31maps.config.path('sfr100.mosaic'), header=True)
fuvint_map = astropy.io.fits.getdata(m31maps.config.path('mod_fuv_int.mosaic'))
fuvred_map = astropy.io.fits.getdata(m31maps.config.path('mod_fuv_red.mosaic'))
nuvint_map = astropy.io.fits.getdata(m31maps.config.path('mod_nuv_int.mosaic'))
nuvred_map = astropy.io.fits.getdata(m31maps.config.path('mod_nuv_red.mosaic'))


# A_FUV and A_NUV maps
afuv_map = (astrogrid.flux.galex_flux2mag(fuvred_map, 'galex_fuv') -
            astrogrid.flux.galex_flux2mag(fuvint_map, 'galex_fuv'))
anuv_map = (astrogrid.flux.galex_flux2mag(nuvred_map, 'galex_nuv') -
            astrogrid.flux.galex_flux2mag(nuvint_map, 'galex_nuv'))


# Write
dest = '/Users/alexialewis/research/phat/uvflux'
hdu = astropy.io.fits.PrimaryHDU(afuv_map, header=hdr)
hdu.writeto(os.path.join(dest, 'A_fuv.fits'), clobber=True)
hdu = astropy.io.fits.PrimaryHDU(anuv_map, header=hdr)
hdu.writeto(os.path.join(dest, 'A_nuv.fits'), clobber=True)
