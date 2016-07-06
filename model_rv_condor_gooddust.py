import numpy as np
import fsps
import emcee
import h5py
import time
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sedpy import attenuation, observate
import compile_data
#import bursty_sfh
from dust import redden

from joblib import Parallel, delayed

from pdb import set_trace


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('reg', type=int, help='region number')
    return parser.parse_args()


def get_data(res='90', dust_curve='cardelli'):
    """
    Gather the GALEX and synthetic UV data. Also get SFR and optical dust.
    Returns FUV flux ratio, NUV flux ratio, delta UV color, and optical Av+dAv from the SFHs.
    """
    fuvdata, nuvdata, otherdata = compile_data.gather_map_data(res, dust_curve)
    sfhcube, sfhcube_upper, sfhcube_lower,sfhhdr = compile_data.gather_sfh(res, sfhcube='sfr_evo_cube_alltimes.fits')

    data_fuv = fuvdata['fluxobs'] / fuvdata['fluxmodint']
    data_fuv = data_fuv[np.isfinite(data_fuv)]
    selbad = np.where(~np.isfinite(data_fuv))

    data_nuv = nuvdata['fluxobs'] / nuvdata['fluxmodint']
    data_nuv = data_nuv[np.isfinite(data_nuv)]

    data_color = (fuvdata['magobs'] - nuvdata['magobs']) - (fuvdata['magmodint'] - nuvdata['magmodint'])
    data_color = data_color[np.isfinite(data_color)]

    av = otherdata['av']
    av = av[np.isfinite(av)]

    dav = otherdata['dav']
    dav = dav[np.isfinite(dav)]

    sfr = otherdata['sfr100']
    sfr = sfr[np.isfinite(sfr)]
    sel = (sfr > 1e-5) & (data_color < 2.)

    sfh = []
    for i in range(len(sfhcube.shape[0])):
        x = sfhcube[i,:,:]
        x[selbad] = np.nan
        sfh.append(x[np.isfinite(x)])
    sfh = np.asarray(sfh)

    return data_fuv, data_nuv, data_color, av, dav, sfh


def ext_func(rv, av, dav, f_bump=1., att=attenuation.conroy):
    """
    Given an R_V and f_bump value, returns the flux ratio or delta color from a specific attenuation curve.

    Parameters
    ----------
    rv : float ; an R_V value
    f_bump : float, optional; strength of the 2175 \AA bump in fraction of MW bump strength
    att : sedpy.attenuation funcion, optional; attenuation curve to use. Default: attenuation.conroy
    """
    specred, lir = redden(wave, spec, av=av, dav=dav,dust_curve=att, nsplit=30)
    #tau_v = av / 1.086
    #tau_lambda = att(wave, R_v=rv, f_bump=f_bump, tau_v=tau_v)
    #f2 = s * np.exp(-tau_lambda)
    mags_red = observate.getSED(wave, specred, filters)
    mags = observate.getSED(wave, spec, filters)

    fluxes_red = [3631*10**(-0.4 * (mag + M31_DM)) for mag in mags_red]
    fluxes = [3631 * 10**(-0.4 * (mag + M31_DM)) for mag in mags]
    val_fuv = fluxes_red[0] / fluxes[0]
    val_nuv = fluxes_red[1] / fluxes[1]

    return val_fuv, val_nuv


def lnlike(data_fuv, data_nuv, sigma_fuv, sigma_nuv, theta, best_av, best_dav):
    """
    data_fuv, etc are for a SINGLE pixel
    """
    model = ext_func(theta[0], best_av, best_dav, f_bump=theta[1], att=ATT)
    val = ((model[0] - data_fuv)**2/sigma_fuv**2) + ((model[1] - data_nuv)**2/sigma_nuv**2)
    return -0.5 * val


def lnprior(theta):
    """
    Set the priors on the model parameters
    """
    # theta = [Rv, av, f_bump] ## doing R_V only for now
    if 0. < theta[0] < 10. and 0. < theta[1] < 1.5:
        return 0.0
    return -np.inf


def lnprob(theta, data_fuv, data_nuv, sigma_fuv, sigma_nuv, best_av, best_dav):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(data_fuv, data_nuv, sigma_fuv, sigma_nuv, theta, best_av, best_dav)


def initialize(init, ndim, nwalkers):
    """
    Offset the initial guess slightly for each walker
    """
    pos = [init + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    return pos


def run_emcee(sampler, run_steps, restart_steps, pos, ndim, nwalkers, n_restarts=4):
    """
    Run an MCMC chain.

    Parameters
    ----------
    sampler : emcee.ensemble.EnsembleSampler
    run_steps : int ; number of steps for each walker takes in the main runs
    restart_steps : int; number of steps each walker takes in the restart runs
    pos : list ; ndim elements specifying starting points of theta
    ndim : int ; number of parameters to fit for
    nwalkers : number of walkers

    Returns
    -------
    sampler : emcee.ensemble.EnsembleSampler
    pos : final position of each walker in theta space

    """

    #print('First burn in')
    pos, lp, state = sampler.run_mcmc(pos, run_steps)

    # continue to burn in, restarting at maximum likelihood position each time
    for i in range(n_restarts):
        #print('Burn in: Restart ' + str(i + 1) + '/' + str(n_restarts))
        sampler.reset()
        pos, lp, state = sampler.run_mcmc(pos, restart_steps)
        sel = np.where(sampler.flatlnprobability == np.max(sampler.flatlnprobability))
        pos = initialize(np.mean(sampler.flatchain[sel], axis=0), ndim, nwalkers)

    # one last burn in run
    #print('Final burn in')
    sampler.reset()
    pos, lp, state = sampler.run_mcmc(pos, run_steps)
    sel = np.where(sampler.flatlnprobability == np.max(sampler.flatlnprobability))
    pos = initialize(np.mean(sampler.flatchain[sel], axis=0), ndim, nwalkers)
    sampler.reset()

    # actual mcmc run
    #print('Actual Run')
    sampler.run_mcmc(pos, run_steps)

    return sampler, pos


def plot_walkers(sampler, nwalkers, ndim, labels=None):
    naxes = ndim + 1
    fig, ax = plt.subplots(1, naxes, figsize=(naxes*3,naxes*1.5))

    pp = np.asarray([sampler.chain[:,:, d] for d in range(ndim)])
    for aa in range(ndim):
        for w in range(nwalkers):
            ax[aa].plot(pp[aa, w, :])
    for w in range(nwalkers):
        ax[aa+1].plot(sampler.lnprobability[w,:])

    for i, ax in enumerate(fig.axes):
        if labels:
            labels.append('ln(prob)')
            ax.set_ylabel(labels[i])
            ax.set_xlabel('Steps')
        ax.tick_params(labelsize=13)
    plt.subplots_adjust(wspace=0.35, right=0.99, top=0.92, left=0.08,
                        bottom=0.13)


def plot_triangle(sampler, labels=None, truths=None, ndim=None):
    import corner
    corner.ScalarFormatter(useOffset=False)
    lim = [(0.9995 * np.nanmin(sampler.flatchain[:,i]), 1.0005 * np.nanmax(sampler.flatchain[:,i])) for i in range(sampler.flatchain.shape[1])]
    fig = corner.corner(sampler.flatchain, truths=truths, labels=labels,
                        range=lim)


def plot_data_dist(datax, datay, sampler):
    from dustvar_paper_figs import make_2dhist_func
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # first plot the real data
    zticks = [1, 2, 5, 10, 20, 50]
    xlim = [-0.01, 1.55]
    ylim = [-1.02, 1.42]
    cnorm = mcolors.LogNorm(vmin=1, vmax=zticks[-1])
    hist_kwargs = {'func': 'count', 'bins': 75, 'xlim': xlim,
                   'ylim': ylim, 'cnorm': cnorm}
    im, cnorm, bindata = make_2dhist_func(ax, datax, datay, datay, cmap=plt.cm.Greys_r, **hist_kwargs)
    ax.grid()
    #plt.plot(datax, datay, marker='o', ms=1.5, color='k', lw=0)

    ## Now over plot draws from the model
    modx = np.linspace(0, 2.5, 100)
    mean = np.percentile(sampler.flatchain[:,0], 50)
    sigma = np.percentile(sampler.flatchain[:,1], 50)
    mu_draw = np.random.normal(mean, sigma, 500)
    for mu in mu_draw:
        mody = modx * ext_func(mu, ftype=FTYPE, band=BAND, att=ATT)
        ax.plot(modx, mody, lw=0.5, color='red', alpha=0.1)

    ax.set_xlabel(r'$A_V$ + 1/2 $dA_V$')
    ax.set_ylabel(r'$\Delta$ UV Color')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



def main(i, **kwargs):

    ## location to store data
    #data_loc = '/Users/alexialewis/research/PHAT/dustvar/'
    data_loc = '/astro/store/phat/arlewis/dustvar/'

    # gather the real data
    y_fuv, y_nuv, y_color, av, dav = get_data()
    y_fuv, y_nuv, y_color = y_fuv, y_nuv, y_color
    z = len(str(len(y_fuv)))

    sigma_fuv, sigma_nuv = 0.3 * y_fuv, 0.3 * y_nuv


    # steps to take in the burn in runs, restarts, and final run
    restart_steps = 500
    run_steps = 1000
    n_restarts = 8

    #initial guess of mu_rv and sigma_rv
    first_init = [2.5, 0.9]
    labels = ['$R_V$', r'$f_{\textrm{bump}}$']
    labs = ['R_V', 'f_bump']

    # number of dimensions and number of walkers
    ndim, nwalkers = len(first_init), 16


    # initialize the first guess with a slight offset for each walker
    pos = initialize(first_init, ndim, nwalkers)

    #z = len(str(len(y_fuv)))
    # note starting time
    t0 = time.time()

    # args is what is passed to lnprob in addiiton to theta
    # doing one pixel
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(y_fuv[i], y_nuv[i],
                                          sigma_fuv[i], sigma_nuv[i],
                                          av[i], dav[i]))

    # Run emcee
    sampler, pos = run_emcee(sampler, run_steps, restart_steps, pos,
                             ndim, nwalkers, n_restarts=n_restarts)

    # note end time
    t1 = time.time()

    # create a file to store data and write results
    region = 'region_' + str(i+1).zfill(z)
    filename = os.path.join(data_loc, 'newred_data_' + region + '.h5')
    with h5py.File(filename, 'w') as hf:
        g = hf.create_group(region)
        g.create_dataset('sampler_chain', data=sampler.chain)
        g.create_dataset('sampler_flatchain', data=sampler.flatchain)
        g.create_dataset('sampler_lnprob', data=sampler.lnprobability)
        g.create_dataset(labs[0], data=np.percentile(sampler.flatchain[:,0], [16, 50, 84]))
        g.create_dataset(labs[1], data=np.percentile(sampler.flatchain[:,1], [16, 50, 84]))
        g.create_dataset('run_time', data=np.around(t1-t0, 2))


if __name__ == '__main__':
    sps = fsps.StellarPopulation()
    sps.params['sfh'] = 4
    sps.params['const'] = 1.0
    sps.params['imf_type'] = 2
    wave, spec = sps.get_spectrum(tage=1.0, peraa=True)

    M31_DM = 24.47
    ATT = attenuation.conroy

    filters = ['galex_fuv', 'galex_nuv']
    filters = observate.load_filters(filters)

    reg_num = get_args().reg
    kwargs = {'wave': wave, 'spec': spec, 'filters': filters, 'M31_DM': M31_DM,
              'ATT': ATT}

    main(reg_num, **kwargs)
