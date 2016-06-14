import numpy as np
import fsps
import emcee

import matplotlib.pyplot as plt

from m31maps.util import make_counter
from sedpy import attenuation, observate
import compile_data

sps = fsps.StellarPopulation()
sps.params['sfh'] = 4
sps.params['const'] = 1.0
sps.params['imf_type'] = 2
wave, s = sps.get_spectrum(tage=1.0, peraa=True)

M31_DM = 24.47
FTYPE = 'color'
BAND ='fuv'

filters = ['galex_fuv', 'galex_nuv']
filters = observate.load_filters(filters)

def get_data(res='90', dust_curve='cardelli'):
    fuvdata, nuvdata, otherdata = compile_data.gather_map_data(res, dust_curve)
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
        #print (mags_red[0]-mags_red[1], mags[0]-mags[1])
    return val



"""
def lnlike(theta, y, ftype='color', band='fuv', att=attenuation.conroy):
    theta[0] = 3.1
    model = theta[2]
    #model = ext_func(theta[0], theta[1], ftype=ftype, band=band, att=att)  #theta[0]=R_V, theta[1]=f_bump
    #model += theta[2]  # nuiscance parameter for offset from 0,0
    sigma =  theta[3]#(0.8) * y
    return -0.5 * (np.sum(((y - model)**2 / (sigma**2 )) - np.log(2 * np.pi * sigma**2)))
"""

def lnlike(theta, y, ftype='color', band='fuv', att=attenuation.conroy):
    return  np.sum(np.log(return_gauss(y, ext_func(theta[0]), theta[1])))
    #-0.5 * np.sum(((y - theta[0])**2 / (theta[1]**2 )) - np.log(2 * np.pi * theta[1]**2))


def lnprior(theta):
    #if 1 < theta[0] < 5 and 0 < theta[1] <= 1.5 and -0.5 < theta[2] < 0.5 and 0. < theta[3] < 20.:
    if 0. < theta[0] < 5. and 0 < theta[1] < 2:
        return 0.0
    return -np.inf


def lnprob(theta, y, ftype, band, att):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    print lnlike(theta, y, ftype=ftype, band=band), theta
    return lp + lnlike(theta, y, ftype=ftype, band=band, att=att)


def initialize(init):
    """
    Offset the initial guess slightly for each walker
    """
    pos = [init + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    return pos



def return_gauss(dcolor, mu, sigma):
    return 1./np.sqrt(2 * np.pi * sigma**2) * np.exp(-(dcolor - mu)**2 / (2 * sigma**2))


if __name__ == '__main__':
    plt.close('all')

    # gather the data
    y_fuv, y_nuv, y_color, avdav = get_data()
    y_fuv, y_nuv, y_color = [y_fuv], [y_nuv], y_color

    #yy = np.asarray(y_color)[0,:]
    yy = y_color

    # number of dimensions and number of walkers
    ndim, nwalkers = 2, 32

    #initial guess
    #first_init = [4, 0.9, 0, 0.1]
    first_init = [2.5, 1]

    att = attenuation.cardelli

    pos = initialize(first_init)

    ## args is what is passed to lnprob in addiiton to theta
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(y_color, FTYPE, BAND, att))

    # burn in
    print('First burn in')
    pos, lp, state = sampler.run_mcmc(pos, 1000)
    n_restart = 4

    # continue to burn in, restarting at maximum likelihood position each time

    for i in range(n_restart):
        #prefix = 'Burn in restart: '
        #c = make_counter(prefix=prefix, end='  done')
        print('Burn in: Restart ' + str(i + 1) + '/' + str(n_restart))
        sampler.reset()
        pos, lp, state = sampler.run_mcmc(pos, 500)
        sel = np.where(sampler.flatlnprobability == np.max(sampler.flatlnprobability))
        pos = initialize(np.mean(sampler.flatchain[sel], axis=0))

    # final burn in
    print('Final burn in')
    sampler.reset()
    sampler.run_mcmc(pos, 1000)
    pos = initialize(np.mean(sampler.flatchain[sel], axis=0))
    sampler.reset()

    # actual mcmc run
    print('Actual Run')
    sampler.run_mcmc(pos, 1000)

    # print out values of final distribution
    print(' ')
    print 'Rv: ', np.percentile(sampler.flatchain[:,0], [16, 50, 84])
    print 'sigma: ', np.percentile(sampler.flatchain[:,1], [16, 50, 84])
    #print '$R_V$: ', np.percentile(sampler.flatchain[:,0], [16, 50, 84])
    #print '$f_{bump}$: ', np.percentile(sampler.flatchain[:,1], [16, 50, 84])
    #print 'C: ', np.percentile(sampler.flatchain[:,2], [16, 50, 84])
    #print 'sigma: ', np.percentile(sampler.flatchain[:,3], [16, 50, 84])
    print(' ')

    ### PLOTTING ###
    #--------------#
    fig = plt.figure()

    n, bins, edges = plt.hist(y_color, bins=100, normed=True)
    dcolor = np.arange(-2, 5, 0.1)
    plt.plot(dcolor, return_gauss(dcolor, ext_func(np.percentile(sampler.flatchain[:,0], 50)), np.percentile(sampler.flatchain[:,1], 50)))
    plt.show()
"""
    # plots the movement of each walker with each step
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,8))
    for w in range(nwalkers):
        ax1.plot(sampler.chain[w,:,0])
        ax2.plot(sampler.chain[w,:,2])
        ax3.plot(sampler.chain[w,:,3])
        ax4.plot(sampler.lnprobability)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)


    # triangle plot showing pdfs
    import corner
    #fig2 = corner.corner(sampler.flatchain, labels=['$R_V$', '$f_{bump}$', 'C'], truths=first_init)


    # plot draws from the distriubtion on top of the data
    fig3 = plt.figure()
    plt.plot(avdav, yy, marker='o', ms=1.5, color='k', lw=0)

    #av = np.linspace(0, 2.5, 100)
    #for rr, ff, cc, ss in sampler.flatchain[np.random.randint(len(sampler.flatchain), size=100)]:
    #   print rr, ff, ext_func(rr, ff, ftype=FTYPE, band=BAND, att=att)
    #    plt.plot(av, av * (ext_func(rr, ff, ftype=FTYPE, band=BAND, att=att)) + cc , lw=0.5, color='red', alpha=0.3)


    plt.show()

"""
