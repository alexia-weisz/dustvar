import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import random
from scipy.misc import logsumexp
import time
import emcee
import compile_data
from pdb import set_trace


def evaluate_lneta(theta, grid):
    mean = theta[0] #+ theta[2] * np.log10(sfr / np.mean(sfr))
    var = theta[1]
    return -0.5 * (grid - mean)**2 / var - 0.5 * np.log(2 * np.pi * var)


def lnlike(theta, grid):
    return np.sum(logsumexp(evaluate_lneta(theta, grid), axis=1))


def lnprior(theta, gridtype='rv'):
    if gridtype == 'rv':
        if 0. < theta[0] < 6. and 0. < theta[1] < 2.:
            return 0.0
    elif gridtype == 'fbump':
        if 0. < theta[0] < 1.5 and 0. < theta[1] < 2.:
            return 0.0
    return -np.inf


def lnprob(theta, grid, gridtype='rv'):
    lp = lnprior(theta, gridtype=gridtype)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, grid)

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
    n_restarts : int (optional) ; number of times to restart the burn-in

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

    return sampler, lp, pos

def plot_walkers(sampler, nwalkers, ndim, labels=None):
    naxes = ndim + 1
    fig, ax = plt.subplots(naxes, 1, figsize=(naxes*1.8,naxes*2.8),sharex=True)
    pp = np.asarray([sampler.chain[:,:, d] for d in range(ndim)])
    for aa in range(ndim):
        for w in range(nwalkers):
            ax[aa].plot(pp[aa, w, 100:])
    for w in range(nwalkers):
        ax[aa+1].plot(sampler.lnprobability[w,100:])

    for i, a in enumerate(fig.axes):
        if labels:
            labels.append('$\ln f$')
            a.set_ylabel(labels[i])
        a.tick_params(labelsize=13)
    ax[-1].set_xlabel('Steps')
    plt.subplots_adjust(hspace=0.12, right=0.96, top=0.92, left=0.15,
                        bottom=0.13)

def plot_triangle(sampler, labels=None, truths=None, ndim=None):
    import corner
    corner.ScalarFormatter(useOffset=False)
    #lim = [(0.9995 * np.nanmin(sampler.flatchain[:,i]), 1.0005 * np.nanmax(sampler.flatchain[:,i])) for i in range(sampler.flatchain.shape[1])]
    fig = corner.corner(sampler.flatchain[100:,:], truths=truths, labels=labels)#,range=lim)


def model(grid, nwalkers, first_init, run_steps, restart_steps, gridtype='rv', n_restarts=0, labels=['$\mu$', '$\sigma$']):

    ndim = len(first_init)

    # initialize the first guess with a slight offset for each walker
    pos = initialize(first_init, ndim, nwalkers)


    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(grid, gridtype))

    # Run emcee
    sampler, lp, pos = run_emcee(sampler, run_steps, restart_steps, pos,
                             ndim, nwalkers, n_restarts=n_restarts)

    t1 = time.time()

    sampler.chain[:,:,1] = np.sqrt(sampler.chain[:,:,1])

    print 'mu: ', np.percentile(sampler.flatchain[:,0], [16, 50, 84])
    print 'sigma: ', np.percentile(sampler.flatchain[:,1], [16,50,84])
    print 'Run took ' + str(np.around(t1-t0, 2))+' seconds.'

    return sampler, lp, pos, np.around(t1-t0, 2)



if __name__ == '__main__':

    selection = False#True
    write = True

    data_loc = '/Users/alexialewis/research/PHAT/dustvar'
    filename = os.path.join(data_loc, 'all_runs_nofbump.h5')

    if selection:
        outfile = os.path.join(data_loc,'final_sampler_rv_nofbump_avdavgt05.h5')
    else:
        outfile = os.path.join(data_loc, 'final_sampler_rv_nofbump.h5')

    nsamples = 50

    fuvdata, nuvdata, otherdata = compile_data.gather_map_data()
    sfr100 = otherdata['sfr100']
    avdav = otherdata['avdav']
    #sel = np.where(sfr100[np.isfinite(sfr100)].flatten() > 1e-6)[0]
    sel = np.where(avdav[np.isfinite(avdav)].flatten() > 0.5)[0]

    with h5py.File(filename, 'r') as hf:
        if selection:
            nregs = len(sel)
            reg_range = sel
        else:
            nregs = len(hf.keys())
            reg_range = range(nregs)

        rvgrid = np.asarray(np.zeros((nregs, nsamples)))
        fbgrid = np.asarray(np.zeros((nregs, nsamples)))

        total_samples = (hf.get(hf.keys()[0]))['sampler_flatchain'].shape[0]

        random.seed(200)
        inds = sorted(random.sample(range(total_samples), nsamples))

        for i, reg in enumerate(reg_range):
            group = hf.get(hf.keys()[reg])
            rvrange = np.asarray(group['sampler_flatchain'][inds,0])
            fbrange = np.asarray(group['sampler_flatchain'][inds,1])

            rvgrid[i,:] = rvrange
            fbgrid[i,:] = fbrange


    # steps to take in the burn in runs, restarts, and final run
    restart_steps = 500
    run_steps = 1000
    n_restarts = 8
    nwalkers = 16

    #initial guess of mu_rv and sigma_rv
    first_init_rv = [2.5, 0.2]
    first_init_fb = [0.8, 0.2]
    ndim = len(first_init_rv)
    labels_rv = ['$\mu_{R_V}$', '$\sigma_{R_V}$']
    labels_fb = ['$\mu_{f_{bump}}$', '$\sigma_{f_{bump}}$']

    sampler_rv, lp_rv, pos_rv, t_rv = model(rvgrid, nwalkers, first_init_rv, run_steps, restart_steps, gridtype='rv', n_restarts=n_restarts, labels=labels_rv)
    sampler_fb, lp_fb, pos_fb, t_fb = model(fbgrid, nwalkers, first_init_fb, run_steps, restart_steps, gridtype='fbump', n_restarts=n_restarts, labels=labels_fb)

    if write:
        #outfile = os.path.join(data_loc, '/final_sampler_rv_fbump.h5')
        rf = h5py.File(outfile, 'w')
        g = rf.create_group('R_V')
        g.create_dataset('sampler_chain', data=sampler_rv.chain)
        g.create_dataset('sampler_flatchain', data=sampler_rv.flatchain)
        g.create_dataset('sampler_lnprob', data=sampler_rv.lnprobability)
        g.create_dataset('mu', data=np.percentile(sampler_rv.flatchain[:,0], [16, 50, 84]))
        g.create_dataset('sigma', data=np.percentile(sampler_rv.flatchain[:,1], [16, 50, 84]))
        g.create_dataset('run_time', data=t_rv)
        g.create_dataset('autocorr_time', data=sampler_rv.acor)

        g = rf.create_group('f_bump')
        g.create_dataset('sampler_chain', data=sampler_fb.chain)
        g.create_dataset('sampler_flatchain', data=sampler_fb.flatchain)
        g.create_dataset('sampler_lnprob', data=sampler_fb.lnprobability)
        g.create_dataset('mu', data=np.percentile(sampler_fb.flatchain[:,0], [16, 50, 84]))
        g.create_dataset('sigma', data=np.percentile(sampler_fb.flatchain[:,1], [16, 50, 84]))
        g.create_dataset('run_time', data=t_fb)
        g.create_dataset('autocorr_time', data=sampler_fb.acor)
        rf.close()


    plot_walkers(sampler_rv, nwalkers, ndim, labels=labels_rv)
    plotname_rvw = os.path.join(data_loc, 'plots', 'walkers_rv.pdf')
    #plt.savefig(plotname_rvw)

    plot_walkers(sampler_fb, nwalkers, ndim, labels=labels_fb)
    plotname_fbw = os.path.join(data_loc, 'plots', 'walkers_fb.pdf')
    #plt.savefig(plotname_fbw)

    plot_triangle(sampler_rv, labels=labels_rv, truths=None, ndim=None)
    plotname_rvt = os.path.join(data_loc, 'plots', 'triangle_rv.pdf')
    #plt.savefig(plotname_rvt)

    plot_triangle(sampler_fb, labels=labels_fb, truths=None, ndim=None)
    plotname_fbt = os.path.join(data_loc, 'plots', 'triangle_fb.pdf')
    #plt.savefig(plotname_fbt)
    plt.show()


