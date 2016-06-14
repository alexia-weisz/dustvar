import numpy as np
import matplotlib.pyplot as plt
from sedpy import attenuation as att
from sedpy import observate
import fsps
from pdb import set_trace


def create_spectrum():
    sps = fsps.StellarPopulation()

    sps.params['sfh'] = 4
    sps.params['const'] = 1.0

    #wave, s = sps.get_spectrum(tage=1.0, peraa=True)
    wave, s = sps.get_spectrum(tage=0.2, peraa=True)

    return wave, s


def make_curve(wave, s, laws, lawnames, Rvs, bumps, filters):
    e_fn = {}
    e_fn_flux = {}
    for law, name in zip(laws, lawnames):
        e_fn[name] = {}
        e_fn_flux[name] = {}
        for rv in Rvs:
            e_fn[name][str(rv)] = {}
            e_fn_flux[name][str(rv)] = {}
            for f_bump in bumps:
                ext = law(wave, R_v=rv, f_bump=f_bump, tau_v=1.0)
                f2 = s * np.exp(-ext)
                mags_red = observate.getSED(wave, f2, filters)
                mags = observate.getSED(wave, s, filters)
                e_fn[name][str(rv)][str(f_bump)] = ((mags_red[0] -
                                                     mags_red[1]) -
                                                    (mags[0] - mags[1]))
                e_fn_flux[name][str(rv)][str(f_bump)] = np.log10(np.sum(s) / np.sum(f2))

    return e_fn, e_fn_flux


def plot_extcurve(ax, lw=2, plot_type='color', legend=False, laws=[att.cardelli, att.smc, att.calzetti, att.conroy], lawnames = ['MW', 'SMC', 'Calz', 'C10'], filters = ['galex_FUV', 'galex_NUV'], plot_laws=[('MW', '3.1', '1.0')], cmap='Spectral'):

    wave, s = create_spectrum()

    Rvs = np.arange(2.2, 4.4, 0.3)
    bumps = np.arange(0, 1.2, 0.2)

    filters = observate.load_filters(filters)

    e_fn_col, e_fn_flux = make_extcurve(wave,s,laws,lawnames,Rvs,bumps,filters)
    if plot_type == 'color':
        e_fn = e_fn_col
    elif plot_type == 'flux':
        e_fn = e_fn_flux

    n_lines = len(plot_laws)

    # set up the color scheme
    cm = plt.get_cmap(cmap)
    color = [cm(1.0 * i / n_lines) for i in range(n_lines)]
    ax.set_color_cycle(color)

    av = np.linspace(0, 2.5, 100)
    label = r'{0} $R_V={1}$, $f_{{bump}}={2}$'

    for pars in plot_laws:
        x = av
        y = av * e_fn[pars[0]][pars[1]][pars[2]]
        ax.plot(x, y, label=label.format(*pars), lw=lw)

    if legend:
        ax.legend(loc=6, bbox_to_anchor=(1.05, 0.58), fontsize=12)
