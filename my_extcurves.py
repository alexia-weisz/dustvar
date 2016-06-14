import numpy as np
import matplotlib.pyplot as pl
from sedpy import attenuation as att
from sedpy import observate
import fsps
from mpltools import color
import astrogrid

sps = fsps.StellarPopulation()

sps.params['sfh'] = 4
sps.params['const'] = 1.0

wave, s = sps.get_spectrum(tage=1.0, peraa=True)

laws = [att.cardelli, att.smc, att.calzetti, att.conroy]
lawnames = ['MW', 'SMC', 'Calz', 'C10']
Rvs = np.arange(2.2, 4.3, 0.3)
bumps = np.arange(0, 1.2, 0.2)

filters = ['galex_FUV', 'galex_NUV']
filters = observate.load_filters(filter_names)

e_fn_col = {}
e_fn_flux_fuv = {}
e_fn_flux_nuv = {}
for law, name in zip(laws, lawnames):
    e_fn_col[name] = {}
    e_fn_flux_fuv[name] = {}
    e_fn_flux_nuv[name] = {}
    for rv in Rvs:
        e_fn_col[name][str(rv)] = {}
        e_fn_flux_fuv[name][str(rv)] = {}
        e_fn_flux_nuv[name][str(rv)] = {}
        for f_bump in bumps:
            ext = law(wave, R_v=rv, f_bump=f_bump)
            f2 = s * np.exp(-ext)
            mags_red = observate.getSED(wave, f2, filters)
            mags = observate.getSED(wave, s, filters)
            fluxes_red = [astrogrid.flux.galex_mag2flux(mags_red[i],filters[i].nick) for i in range(len(filters))]
            fluxes = [astrogrid.flux.galex_mag2flux(mags[i],filters[i].nick) for i in range(len(filters))]
            e_fn_col[name][str(rv)][str(f_bump)] = (mags_red[0] - mags_red[1]) - (mags[0] - mags[1])
            #e_fn_flux[name][str(rv)][str(f_bump)] = np.log10(s/f2)
            e_fn_flux_fuv[name][str(rv)][str(f_bump)] = np.log10(fluxes[0]/fluxes_red[0])
            e_fn_flux_nuv[name][str(rv)][str(f_bump)] = np.log10(fluxes[1]/fluxes_red[1])


plot_laws = [('MW', '3.1', '1.0'),
             ('MW', '2.2', '1.0'),
             ('MW', '4.0', '1.0'),
             ('SMC','3.1', '0.0'),
             ('Calz', '4.0', '0.0'),
             ('C10', '3.1', '1.0'),
             ('C10', '3.1', '0.6'),
             ('C10', '3.1', '0.2'),
             ]

n_lines = len(plot_laws)
color.cycle_cmap(n_lines)
color.cycle_cmap(n_lines, cmap='Spectral')

av = np.linspace(0, 2.2, 100)
label = r'{0} $R_V={1}$, $f_{{bump}}={2}$'
fig, axes = pl.subplots()
for pars in plot_laws:
    axes.plot(av, av* e_fn_col[pars[0]][pars[1]][pars[2]], label=label.format(*pars))
axes.set_ylabel(r'$\Delta (FUV-NUV)$')
axes.set_xlabel(r'$A_V$')
axes.legend(loc=0)
fig.show()


