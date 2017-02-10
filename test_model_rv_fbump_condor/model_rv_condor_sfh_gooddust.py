import numpy as np
import fsps
import emcee
import h5py
import time
import os
import sys
import astrogrid

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import bursty_sfh
import astropy.coordinates
from sedpy import attenuation, observate
import compile_data

from joblib import Parallel, delayed

from pdb import set_trace

CURRENT_SP = []
DIST = astropy.coordinates.Distance(distmod=24.47)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('reg', type=int, help='region number')
    return parser.parse_args()


def get_data(ind, res='90', dust_curve='cardelli'):
    """
    Gather the GALEX and synthetic UV data. Also get SFR and optical dust.
    Returns FUV flux ratio, NUV flux ratio, delta UV color, and optical Av+dAv from the SFHs.
    """
    fuvdata, nuvdata, otherdata = compile_data.gather_map_data(res, dust_curve)

    data_fuv = fuvdata['fluxobs'] / fuvdata['fluxmodint']
    selgood = np.isfinite(data_fuv)
    data_fuv = data_fuv[np.isfinite(data_fuv)]

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

    return data_fuv[ind], data_nuv[ind], data_color[ind], av[ind], dav[ind], len(dav)


def get_sfh_metals(ind, res='90', dust_curve='cardelli'):
    fsps_kwargs = {'imf_type': astrogrid.flux.IMF_TYPE['Kroupa']}

    fuvdata, nuvdata, otherdata = compile_data.gather_map_data(res, dust_curve)
    sfhcube, sfhcube_upper, sfhcube_lower, sfhhdr, metalcube = compile_data.gather_sfh(res, sfhcube='sfr_evo_cube_alltimes.fits',metalcube='metal_evo_cube.fits')

    data_fuv = fuvdata['fluxobs'] / fuvdata['fluxmodint']
    selgood = np.isfinite(data_fuv)

    sfh, metals = [], []
    for i in range(sfhcube.shape[0]):
        x = sfhcube[i,:,:]
        sfh.append(x[selgood])
        y = metalcube[i,:,:]
        metals.append(y[selgood])
    sfh = np.asarray(sfh)[:,ind]
    metals = np.asarray(metals)[:,ind]

    t1 = np.arange(6.6, 9.9, 0.1)
    t2 = t1 + 0.1
    t2[-1] = 10.15

    i = sfh > 0
    sfrtime = 8
    j = t1 < sfrtime

    if not np.any(i & j):
        # Use most recent metallicity available, or else assume solar
        logZ = metals[i][0] if np.any(i) else 0
    else:
        # 100 Myr mean value
        logZ = np.log10(np.mean(10**metals[i & j]))
    fsps_kwargs['zmet'] = astrogrid.flux.get_zmet(logZ)

    t1, t2 = 10**t1, 10**t2
    # Rescale 1st age bin
    sfh[0] *= 1. - t1[0]/t2[0]
    t1[0] = 0

    age, sfr = (t1, t2), sfh

    return age, sfr


def calc_sed(sfr, age, **kwargs):
    """Calculate the SED for a binned SFH.
    Parameters
    ----------
    sfr : 1d array_like
        SFR values (Msun yr-1) for the bins in the SFH.
    age : tuple or array_like
        Ages (i.e., lookback times, yr) of the bin edges in the SFH. In the
        tuple form, the first element is a array of ages for the young
        edges of the bins, and the second element gives the ages for the
        old edges. Both arrays have the same length as `sfr`.
        Alternatively, a single array of length ``len(sfr)+1`` may be
        specified for the ages of all of the bin edges. SFH bins are
        assumed to be in order of increasing age, so the youngest bins
        (closest to the present) are listed first and the oldest bins
        (closest to the big bang) are listed last.
    age_observe : float or list, optional
        Age (i.e., lookback times, yr) at which the SED is calculated. May
        also be a list of ages. Default is 1. Note that 0 will throw a
        RuntimeWarning about dividing by zero. It is safer to use a small
        positive number instead (hence 1 yr as the default).
    bin_res : float, optional
        Time resolution factor used for resampling the input SFH. The time
        step in the resampled SFH is equal to the narrowest age bin divided
        by this number. Default is 20.
    av, dav : float, optional
        Foreground and differential V-band extinction parameters. Default
        is None (0). [1]_
    nsplit : int, optional
        Number of pieces in which to split the spectrum. Default is 30. [1]_
    dust_curve : string or function, optional
        The name of a key in the `DUST_TYPE`. Default is 'cardelli'. May
        instead give a function that returns ``A_lambda/A_V`` (extinction
        normalized to the total V-band extinction) for a given array of
        input wavelengths in Angstroms (see the `attenuation` module in
        `sedpy`). [1]_
    fsps_kwargs : dict, optional
        Dictionary of keyword arguments for `fsps.StellarPopulation`.
        Default is an empty dictionary. 'sfh' is always set to 0.

    """

    if len(age) == 2:
        try:
            age = np.append(age[0], age[1][-1])  # One array of bin edges
        except (TypeError, IndexError):
            # Probably not a length-2 sequence of sequences
            pass

    age_list = kwargs.get('age_observe', 0.0001)
    try:
        len_age_list = len(age_list)
    except TypeError:
        # age_list is a single value
        age_list = [age_list]
        len_age_list = 0

    bin_res = kwargs.get('bin_res', 20.0)
    av, dav = kwargs.get('av', None), kwargs.get('dav', None)
    rv, f_bump = kwargs.get('rv', 3.1), kwargs.get('f_bump', 1.0)
    nsplit = kwargs.get('nsplit', 30)
    dust_curve = kwargs.get('dust_curve', 'cardelli')
    if isinstance(dust_curve, basestring):
        dust_curve = DUST_CURVE[dust_curve]
    fsps_kwargs = kwargs.get('fsps_kwargs', {})

    # To save time, create StellarPopulation only when necessary
    try:
        sp = CURRENT_SP[0]
    except IndexError:
        sp = fsps.StellarPopulation()
        CURRENT_SP.append(sp)
    fsps_kwargs['sfh'] = 0
    for key, val in fsps_kwargs.items():
        sp.params[key] = val

    names = ['t1', 't2', 'sfr']
    names = [name.encode('utf-8') for name in names]# unicode names not allowed
    types = [float, float, float]
    dtypes = zip(names, types)
    sfh = np.array(zip(age[:-1], age[1:], sfr), dtypes)
    # Resample the SFH to a high time resolution
    age, sfr = bursty_sfh.burst_sfh(f_burst=0, sfh=sfh, bin_res=bin_res)[:2]

    output = bursty_sps(age, sfr, sp, lookback_time=age_list, av=av, dav=dav, npslit=nsplit, dust_curve=dust_curve, rv=rv, f_bump=f_bump)

    wave, spec, mstar, lum_ir = output

    if not len_age_list:
        spec, mstar, lum_ir = spec[0], mstar, lum_ir

    return wave, spec, lum_ir



def bursty_sps(lt, sfr, sps, lookback_time=[0.0001], dust_curve=attenuation.conroy, rv=3.1, f_bump=1.0, av=None, dav=None, nsplit=9, logzsol=None, **extras):
    sps.params['sfh'] = 0  # make sure SSPs
    ssp_ages = 10**sps.ssp_ages  # in yrs
    if logzsol is None:
        wave, spec = sps.get_spectrum(peraa=True, tage=0)
        mass = sps.stellar_mass.copy()
    else:
        assert(sps._zcontinuous > 0)
        spec, mass = [], []
        for tage, logz in zip(ssp_ages/1e9, logzsol):
            sps.params['logzsol'] = logz
            spec.append(sps.get_spectrum(peraa=True, tage=tage)[1])
            mass.append(sps.stellar_mass)
        spec = np.array(spec)
        mass = np.array(mass)
        wave = sps.wavelengths

    # Redden the SSP spectra
    spec, lir = redden(wave, spec, rv=rv, f_bump=f_bump, av=av, dav=dav,
                       dust_curve=dust_curve, nsplit=nsplit)

    # Get interpolation weights based on the SFH
    target_lt = np.atleast_1d(lookback_time)
    aw = bursty_sfh.sfh_weights(lt, sfr, ssp_ages, lookback_time=target_lt, **extras)

    # Do the linear combination
    int_spec = (spec[None,:,:] * aw[:,:,None]).sum(axis=1)
    mstar = (mass[None,:] * aw).sum(axis=-1)
    if lir is not None:
        lir_tot = (lir[None,:] * aw).sum(axis = -1)
    else:
        lir_tot = 0

    return wave, int_spec, mstar, lir_tot


def redden(wave, spec, rv=3.1, f_bump=1.0, av=None, dav=None, nsplit=9, dust_curve=None, wlo=1216., whi=2e4, **kwargs):
    """
    from scombine.dust
    """
    if (av is None) and (dav is None):
        return spec, None
    if dust_curve is None:
        print('Warning:  no dust curve was given')
        return spec, None
    #only split if there's a nonzero dAv
    nsplit = nsplit * np.any(dav > 0) + 1
    lisplit = spec/nsplit
    # Enable broadcasting if av and dav aren't vectors
    # and convert to an optical depth instead of an attenuation

    av = np.atleast_1d(av)/1.086
    dav = np.atleast_1d(dav)/1.086
    lisplit = np.atleast_2d(lisplit)
    #uniform distribution from Av to Av + dAv
    avdist = av[None, :] + dav[None,:] * ((np.arange(nsplit) + 0.5)/nsplit)[:,None]
    #apply it
    ee = (np.exp(-dust_curve(wave, R_v=rv, f_bump=f_bump)[None,None,:] * avdist[:,:,None]))
    spec_red = (ee * lisplit[None,:,:]).sum(axis = 0)
    #get the integral of the attenuated light in the optical-
    # NIR region of the spectrum
    opt = (wave >= wlo) & (wave <= whi)
    lir = np.trapz((spec - spec_red)[:,opt], wave[opt], axis = -1)
    return np.squeeze(spec_red), lir


def ext_func(sfr, age, rv, av, dav, f_bump=1., att=attenuation.conroy):
    """
    Given an R_V and f_bump value, returns the flux ratio or delta color from a specific attenuation curve.

    Parameters
    ----------
    rv : float ; an R_V value
    f_bump : float, optional; strength of the 2175 \AA bump in fraction of MW bump strength
    att : sedpy.attenuation funcion, optional; attenuation curve to use. Default: attenuation.conroy
    """


    wave_red, spec_red, lum_ir_red = calc_sed(sfr, age, av=av, dav=dav, rv=rv, f_bump=f_bump, dust_curve=att, nsplit=30)
    wave, spec, lum_ir = calc_sed(sfr, age, rv=rv, f_bump=f_bump, dust_curve=att, nsplit=30)

    mags_red = astrogrid.flux.calc_mag(
            wave_red, spec_red, bands, dmod=DIST.distmod)
    mags = astrogrid.flux.calc_mag(wave, spec, bands, dmod=DIST.distmod)

    fluxes_red = [astrogrid.flux.mag2flux(mags_red[0], bands[0]), astrogrid.flux.mag2flux(mags_red[1], bands[1])]
    fluxes = [astrogrid.flux.mag2flux(mags[0], bands[0]), astrogrid.flux.mag2flux(mags[1], bands[1])]

    val_fuv = fluxes_red[0] / fluxes[0]
    val_nuv = fluxes_red[1] / fluxes[1]

    return val_fuv, val_nuv


def lnlike(data_fuv, data_nuv, sigma_fuv, sigma_nuv, theta, best_av, best_dav, age, sfr):
    """
    data_fuv, etc are for a SINGLE pixel
    """
    model = ext_func(sfr, age, theta[0], best_av, best_dav, f_bump=theta[1], att=ATT)
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


def lnprob(theta, data_fuv, data_nuv, sigma_fuv, sigma_nuv, best_av, best_dav, age, sfr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(data_fuv, data_nuv, sigma_fuv, sigma_nuv, theta, best_av, best_dav, age, sfr)


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

    # gather the real data for region i
    y_fuv, y_nuv, y_color, av, dav, z = get_data(i)
    sigma_fuv, sigma_nuv = 0.3 * y_fuv, 0.3 * y_nuv

    # get the sfh info
    age, sfr = get_sfh_metals(i)

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
                                    args=(y_fuv, y_nuv, sigma_fuv, sigma_nuv,
                                          av, dav, age, sfr))

    # Run emcee
    sampler, pos = run_emcee(sampler, run_steps, restart_steps, pos,
                             ndim, nwalkers, n_restarts=n_restarts)

    # note end time
    t1 = time.time()

    # create a file to store data and write results
    region = 'region_' + str(i+1).zfill(z)
    filename = os.path.join(data_loc, 'newred_sfh_data_' + region + '.h5')
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

    bands = ['galex_fuv', 'galex_nuv']
    filters = observate.load_filters(bands)

    reg_num = get_args().reg
    kwargs = {'wave': wave, 'spec': spec, 'filters': filters, 'M31_DM': M31_DM,
              'ATT': ATT}

    main(reg_num, **kwargs)
