import numpy as np
import fsps
import emcee

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sedpy import attenuation, observate
import compile_data

from pdb import set_trace


sps = fsps.StellarPopulation()
sps.params['sfh'] = 4
sps.params['const'] = 1.0
sps.params['imf_type'] = 2
wave, s = sps.get_spectrum(tage=1.0, peraa=True)

M31_DM = 24.47
FTYPE = 'color'
BAND ='fuv'
ATT = attenuation.cardelli

filters = ['galex_fuv', 'galex_nuv']
filters = observate.load_filters(filters)

def get_data(res='90', dust_curve='cardelli'):
    """
    Gather the GALEX and synthetic UV data. Also get SFR and optical dust.
    Returns FUV flux ratio, NUV flux ratio, delta UV color, and optical Av+dAv from the SFHs.
    """
    fuvdata, nuvdata, otherdata = compile_data.gather_map_data(res=res, dust_curve=dust_curve)
    data_fuv = fuvdata['fluxobs'] / fuvdata['fluxmodint']
    data_fuv = data_fuv[np.isfinite(data_fuv)]

    data_nuv = nuvdata['fluxobs'] / nuvdata['fluxmodint']
    data_nuv = data_nuv[np.isfinite(data_nuv)]

    data_color = (fuvdata['magobs'] - nuvdata['magobs']) - (fuvdata['magmodint'] - nuvdata['magmodint'])
    data_color = data_color[np.isfinite(data_color)]

    av = otherdata['avdav']
    av = av[np.isfinite(av)]

    sfr = otherdata['sfr100']
    sfr = sfr[np.isfinite(sfr)]
    sel = (sfr > 1e-5) & (data_color < 2.)

    return data_fuv[sel], data_nuv[sel], data_color[sel], av[sel]


def ext_func(rv, f_bump=1., ftype='color', band='fuv', att=attenuation.conroy):
    """
    Given an R_V and f_bump value, returns the flux ratio or delta color from a specific attenuation curve.

    Parameters
    ----------
    rv : float ; an R_V value
    f_bump : float, optional; strength of the 2175 \AA bump in fraction of MW bump strength
    ftype : string, optional; ['color', 'flux'] - type of data to return. either the flux ratio or delta UV color. Default: 'color'
    band : string, optional; ['fuv', 'nuv'] -- band to use when ftype='flux'. Default: 'fuv'
    att : sedpy.attenuation funcion, optional; attenuation curve to use. Default: attenuation.conroy
    """
    tau_lambda = att(wave, R_v=rv, f_bump=f_bump, tau_v=1.0)
    f2 = s * np.exp(-tau_lambda)
    mags_red = observate.getSED(wave, f2, filters)
    mags = observate.getSED(wave, s, filters)
    if ftype == 'flux':
        fluxes_red = [3631*10**(-0.4 * (mag_red+M31_DM)) for mag in [mags_red]]
        fluxes = [3631 * 10**(-0.4 * (mag + M31_DM)) for mag in [mags]]
        ind = 0 if band == 'fuv' else 1
        val = fluxes_red[ind] / fluxes[ind]
    elif ftype == 'color':
        val = (mags_red[0] - mags_red[1]) - (mags[0] - mags[1])
    return val

def lnlike(rvweights, data):
    """
    Calculate the distances between the data pdfs and the rv-based model pdfs
    Returns the sum of the distances between the data and model at all points on the R_V grid.
    """
    val = np.sum((rvweights - data)**2, axis=1)
    return -0.5 * np.sum(val)


def lnprior(theta):
    """
    Set the priors on the model parameters
    """
    # theta = [mu_rv, sigma_rv]
    if 0. < theta[0] < 10. and 0 < theta[1] < 2:
        return 0.0
    return -np.inf


def lnprob(theta, data, rvgrid):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(return_rv_weights(rvgrid, theta[0], theta[1]), data)


def initialize(init, ndim, nwalkers):
    """
    Offset the initial guess slightly for each walker
    """
    pos = [init + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    return pos


def return_gauss(data, mu, sigma):
    """
    Returns pdf of gaussian distribution of mean mu and dispersions sigma given data

    Parameters
    ----------
    data : values on which to determine the pdf
    mu : mean of the gaussian distribution
    sigma : standard deviation of the gaussian
    """
    return 1./np.sqrt(2 * np.pi * sigma**2) * np.exp(-(data - mu)**2 / (2 * sigma**2))


def return_rvgrid(rvmin=0.05, rvmax=10., rvdelt=0.05):
    """
    Create grid of R_V values from rvmin to rvmax with spacing of rvdelt.
    """
    rv_grid = np.arange(rvmin, rvmax, rvdelt)
    return rv_grid


def return_uvgrid(rv_grid, att=attenuation.conroy):
    """
    Given R_V grid, returns Delta UV color grid based on an attenuation curve att.
    """
    uv_grid = np.asarray([ext_func(rv, att=ATT) for rv in rv_grid])
    return uv_grid


def return_rv_weights(rvgrid, mean_rv, sig_rv):
    """
    Calculate the gaussian pdf of a mean R_V and a sigma R_V (theta) across a grid of R_V points.
    """
    # weights computed on rv grid to use on the UV grid
    return return_gauss(rvgrid, mean_rv, sig_rv)


def make_data_pdf(mean, error, uvgrid):
    """
    For each data point mean and error, compute the gaussian pdf across uvgrid
    """
    return return_gauss(uvgrid, mean, error)


def run_emcee(sampler, run_steps, restart_steps, pos, ndim, nwalkers):
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

    print('First burn in')
    pos, lp, state = sampler.run_mcmc(pos, run_steps)
    n_restart = 4

    # continue to burn in, restarting at maximum likelihood position each time
    for i in range(n_restart):
        print('Burn in: Restart ' + str(i + 1) + '/' + str(n_restart))
        sampler.reset()
        pos, lp, state = sampler.run_mcmc(pos, restart_steps)
        sel = np.where(sampler.flatlnprobability == np.max(sampler.flatlnprobability))
        pos = initialize(np.mean(sampler.flatchain[sel], axis=0), ndim, nwalkers)

    # one last burn in run
    print('Final burn in')
    sampler.reset()
    pos, lp, state = sampler.run_mcmc(pos, run_steps)
    sel = np.where(sampler.flatlnprobability == np.max(sampler.flatlnprobability))
    pos = initialize(np.mean(sampler.flatchain[sel], axis=0), ndim, nwalkers)
    sampler.reset()

    # actual mcmc run
    print('Actual Run')
    sampler.run_mcmc(pos, run_steps)

    return sampler, pos


def plot_walkers(sampler, nwalkers, labels=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4))
    for w in range(nwalkers):
        ax1.plot(sampler.chain[w,:,0])
        ax2.plot(sampler.chain[w,:,1])
        ax3.plot(sampler.lnprobability)
    ax1.ticklabel_format(useOffset=False)
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



if __name__ == '__main__':
    plt.close('all')

    # gather the real data
    y_fuv, y_nuv, y_color, avdav = get_data()
    y_fuv, y_nuv, y_color = [y_fuv], [y_nuv], y_color


    # Create a grid of R_V data points
    rvgrid = return_rvgrid()

    # Create a grid of \Delta UV color data points given the grid of R_V data points and an attenuation curve.
    uvgrid = return_uvgrid(rvgrid, att=ATT)


    ## TEST DATA
    #rv_array = np.random.uniform(5., 5.5, 100)
    #temp_mean = return_uvgrid(rv_array)
    #temp_mean = np.random.randn(100)
    #temp_data = make_data_pdf(-0.03, 0.05, uvgrid)
    rv_array = np.random.normal(1.8, 0.12, 1000)
    temp_mean = return_uvgrid(rv_array)
    temp_data = np.asarray([make_data_pdf(tm, tm*0.2, uvgrid) for tm in temp_mean])


    # Given the grid of Delta UV color calculated above, determine the gaussian pdf for each data point of delta UV color, yc, across the grid
    #temp_data = np.asarray([make_data_pdf(yc, yc*0.5, uvgrid) for yc in y_color])


    # number of dimensions and number of walkers
    ndim, nwalkers = 2, 32

    # steps to take in the burn in runs, restarts, and final run
    restart_steps = 500
    run_steps = 1000

    #initial guess of mu_rv and sigma_rv
    first_init = [2.5, 0.5]
    labels = [r'$\mu_{R_V}$', r'$\sigma_{R_V}$']


    # initialize the first guess with a slight offset for each walker
    pos = initialize(first_init, ndim, nwalkers)

    ## args is what is passed to lnprob in addiiton to theta
    ## temp_data are the data pdfs
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(temp_data, rvgrid))

    sampler, pos = run_emcee(sampler, run_steps, restart_steps, pos, ndim, nwalkers)


    # print out values of final distribution
    print 'Rv: ', np.percentile(sampler.flatchain[:,0], [16, 50, 84])
    print 'sigma: ', np.percentile(sampler.flatchain[:,1], [16, 50, 84])
    print(' ')



    ### PLOTTING ###
    #--------------#
    # plot the movement of each walker with each step
    #plot_walkers(sampler, nwalkers, labels=labels)

    # triangle plot showing pdfs
    plot_triangle(sampler, labels=labels, truths=first_init, ndim=ndim)

    # plot draws from the distriubtion on top of the real data
    #plot_data_dist(avdav, y_color, sampler)

    plt.show()
