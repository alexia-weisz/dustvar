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


def lnlike(theta, grid):
    mean1, mean2 = theta[0], theta[1]
    sig1, sig2, sig12 = theta[2], theta[3], theta[4]
    #cov2 = np.asarray(([sig1**2, sig12], [sig12, sig2**2])) #parameterize differently to be positive definite
    ll = np.asarray(([np.exp(sig1), 0], [sig12, np.exp(sig2)]))
    cov = np.dot(ll, ll.T)

    icov = np.linalg.inv(cov)  #better to factorize than to invert
    #s, lndetcov = np.linalg.slogdet(cov)#np.linalg.det(cov)
    #dcov = s * lndetcov
    dcov = np.linalg.det(cov)
    rv = grid[:,:,0]
    fb = grid[:,:,1]
    dp = 0.
    for k in range(rv.shape[0]):
        r = np.asarray((rv[k,:] - mean1, fb[k,:] - mean2))
        dr = np.dot(icov, r)
        ff = np.sum(r * dr, axis=0)
        foo = -0.5 * ff - 0.5 * dcov
        dp += logsumexp(foo)
        #set_trace()
        #dp += logsumexp(-0.5 * np.dot(r.T, np.dot(icov, r)) - 0.5 * dcov)
        #set_trace()
        #dp += -0.5 * np.dot(r.T, np.dot(icov, r)) - 0.5 * np.log(dcov)
    return dp


def lnprior(theta):
    if 0. < theta[0] < 6. and 0. < theta[1] < 1.5 and 0. < theta[2] < 2.0 and 0. < theta[3] < 2.0 and 0. < theta[4] < 2.0:
        return 0.0
    return -np.inf


def lnprob(theta, grid, foo):
    lp = lnprior(theta)
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
    #set_trace()
    pos, lp, state = sampler.run_mcmc(pos, run_steps)

    # continue to burn in, restarting at maximum likelihood position each time
    for i in range(n_restarts):
        #print('Burn in: Restart ' + str(i + 1) + '/' + str(n_restarts))
        sampler.reset()
        #set_trace()
        pos, lp, state = sampler.run_mcmc(pos, restart_steps)
        sel = np.where(sampler.flatlnprobability == np.max(sampler.flatlnprobability))
        pos = initialize(np.mean(sampler.flatchain[sel], axis=0), ndim, nwalkers)

    # one last burn in run
    #print('Final burn in')
    sampler.reset()
    #set_trace()
    pos, lp, state = sampler.run_mcmc(pos, run_steps)
    sel = np.where(sampler.flatlnprobability == np.max(sampler.flatlnprobability))
    pos = initialize(np.mean(sampler.flatchain[sel], axis=0), ndim, nwalkers)
    sampler.reset()

    # actual mcmc run
    #print('Actual Run')
    #set_trace()
    sampler.run_mcmc(pos, run_steps)

    return sampler, lp, pos



def model(grid, nwalkers, first_init, run_steps, restart_steps, n_restarts=0, threads=1, labels=['$\mu$', '$\sigma$']):

    ndim = len(first_init)

    # initialize the first guess with a slight offset for each walker
    pos = initialize(first_init, ndim, nwalkers)

    t0 = time.time()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(grid, None),
                                    threads=threads)

    # Run emcee
    sampler, lp, pos = run_emcee(sampler, run_steps, restart_steps, pos,
                                 ndim, nwalkers, n_restarts=n_restarts)

    t1 = time.time()

    #sampler.chain[:,:,1] = np.sqrt(sampler.chain[:,:,1])

    print 'mu_rv: ', np.percentile(sampler.flatchain[:,0], [16, 50, 84])
    print 'mu_fbump: ', np.percentile(sampler.flatchain[:,1], [16, 50, 84])
    print 'sigma_rv: ', np.percentile(sampler.flatchain[:,2], [16, 50, 84])
    print 'sigma_fbump: ', np.percentile(sampler.flatchain[:,3], [16, 50, 84])
    print 'sigma_rf: ', np.percentile(sampler.flatchain[:,4], [16, 50, 84])
    print 'Run took ' + str(np.around(t1-t0, 2))+' seconds.'

    return sampler, lp, pos, np.around(t1-t0, 2)


def plot_walkers(sampler, nwalkers, ndim, labels=None):
    naxes = ndim + 1
    fig, ax = plt.subplots(naxes, 1, figsize=(8,9), sharex=True)
    #pp = np.asarray([sampler.chain[:,:, d] for d in range(ndim)])
    for aa in range(ndim):
        for w in range(nwalkers):
            ax[aa].plot(sampler.chain[w, :, aa])
            #ax[aa].plot(pp[aa, w, 100:])
    for w in range(nwalkers):
        ax[aa+1].plot(sampler.lnprobability[w, :])

    for i, a in enumerate(fig.axes):
        if labels:
            labels.append('$\ln f$')
            a.set_ylabel(labels[i])
        a.tick_params(labelsize=13)
    ax[-1].set_xlabel('Steps')
    plt.subplots_adjust(hspace=0.12, right=0.96, top=0.98, left=0.15,
                        bottom=0.13)

def plot_triangle(sampler, labels=None, truths=None, ndim=None):
    import corner
    corner.ScalarFormatter(useOffset=False)
    #lim = [(0.9995 * np.nanmin(sampler.flatchain[:,i]), 1.0005 * np.nanmax(sampler.flatchain[:,i])) for i in range(sampler.flatchain.shape[1])]
    fig = corner.corner(sampler.flatchain[100:,:], truths=truths, labels=labels)#,range=lim)

def write_to_file(outfile, sampler, runtime):
     #outfile = os.path.join(data_loc, '/final_sampler_rv_fbump.h5')
    rf = h5py.File(outfile, 'w')
    g = rf.create_group('results')
    g.create_dataset('sampler_chain', data=sampler.chain)
    g.create_dataset('sampler_flatchain', data=sampler.flatchain)
    g.create_dataset('sampler_lnprob', data=sampler.lnprobability)
    g.create_dataset('mu_rv', data=np.percentile(sampler.flatchain[:,0], [16, 50, 84]))
    g.create_dataset('mu_fb', data=np.percentile(sampler.flatchain[:,1], [16, 50, 84]))
    g.create_dataset('sigma_rv', data=np.percentile(sampler.flatchain[:,2], [16, 50, 84]))
    g.create_dataset('sigma_fb', data=np.percentile(sampler.flatchain[:,3], [16, 50, 84]))
    g.create_dataset('sigma_co', data=np.percentile(sampler.flatchain[:,4], [16, 50, 84]))
    g.create_dataset('run_time', data=runtime)
    g.create_dataset('autocorr_time', data=sampler.acor)
    rf.close()




if __name__ == '__main__':

    selection = False
    write = True

    if os.environ['PATH'][1:6] == 'astro':
        _TOP_DIR = '/astro/store/phat/arlewis/'
        threads = 4
    else:
        _TOP_DIR = '/Users/alexialewis/research/PHAT/'
        threads = 1

    data_loc = os.path.join(_TOP_DIR, 'dustvar')
    filename = os.path.join(data_loc, 'all_runs.h5')

    if selection:
        outfile = os.path.join(data_loc,'final_sampler_rv_fbump_cov_avdavgt05.h5')
    else:
        outfile = os.path.join(data_loc, 'final_sampler_rv_fbump_cov.h5')

    nsamples = 50

    fuvdata, nuvdata, otherdata = compile_data.gather_map_data()
    sfr100 = otherdata['sfr100']
    avdav = otherdata['avdav']
    #sel = np.where(sfr100[np.isfinite(sfr100)].flatten() > 1e-6)[0]
    sel = np.where(avdav[np.isfinite(avdav)].flatten() > 1.0)[0]

    print "reading in region pdfs..."
    with h5py.File(filename, 'r') as hf:
        if selection:
            nregs = len(sel)
            reg_range = sel
        else:
            nregs = len(hf.keys())
            reg_range = range(nregs)

        #nregs = 100
        #reg_range = range(nregs)
        grid = np.asarray(np.zeros((nregs, nsamples, 2)))

        total_samples = (hf.get(hf.keys()[0]))['sampler_flatchain'].shape[0]

        random.seed(200)
        inds = sorted(random.sample(range(total_samples), nsamples))

        for i, reg in enumerate(reg_range):
            group = hf.get(hf.keys()[reg])
            rvrange = np.asarray(group['sampler_flatchain'][inds,0])
            fbrange = np.asarray(group['sampler_flatchain'][inds,1])

            grid[i,:,0] = rvrange
            grid[i,:,1] = fbrange


    # steps to take in the burn in runs, restarts, and final run
    restart_steps = 250
    run_steps = 500
    n_restarts = 4
    nwalkers = 64

    #initial guess of mu_rv and sigma_rv
    first_init = [3.1, 0.8, 0.3, 0.3, 0.1]
    ndim = len(first_init)
    labels = ['$\mu_{R_V}$','$\mu_{f_{bump}}$','$\sigma_{R_V}$','$\sigma_{f_{bump}}$', '$\sigma_{R_V, f_{bump}}$']

    print "starting mcmc..."
    sampler, lp, pos, t = model(grid, nwalkers, first_init, run_steps, restart_steps, n_restarts=n_restarts, labels=labels, threads=threads)

    if write:
        write_to_file(outfile, sampler, t)

    plot_loc = data_loc + '/plots/'
    plot_triangle(sampler, labels=labels, truths=None, ndim=None)
    plt.savefig(plot_loc + 'simult_triangle_all.pdf')
    plot_walkers(sampler, nwalkers, ndim, labels=labels)
    plt.savefig(plot_loc + 'simult_walkers_all.pdf')
    #plt.show()


