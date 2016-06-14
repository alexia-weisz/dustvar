import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.stats import norm

from pdb import set_trace


def plot_pdfs(filenames, save=False, regs=None, cumulative=False):
    # R_V limits
    rvmin, rvmax = 0, 11
    rvgrid = np.linspace(rvmin,rvmax, 101)
    x_rv = np.linspace(rvmin, rvmax, 100)

    # f_bump limits
    fbmin, fbmax = 0, 2.0
    fbgrid = np.linspace(fbmin, fbmax, 101)
    x_fb = np.linspace(fbmin, fbmax, 100)

    ps_rv, ps_fb = [], []
    lps_rv, lps_fb = [], []

    for infile in filenames:
        with h5py.File(infile, 'r') as hf:
            if regs is None:
                nregs = len(hf.keys())
            else:
                nregs = regs
            n = (hf.get(hf.keys()[0]))['sampler_flatchain'].shape[0]
            #set_trace()
            fig, ax = plt.subplots(1, 2, figsize=(8,5))
            #plt.subplots_adjust(hspace=0.05, left=0.15, right=0.95, top=0.95, bottom=0.08, wspace=0.2)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.1)

            for i in range(500,501):
                color = next(plt.gca()._get_lines.prop_cycler)
                c = color['color']

                group = hf.get(hf.keys()[i])

                rv_pdf = np.asarray(group['sampler_flatchain'][:,0])
                fb_pdf = np.asarray(group['sampler_flatchain'][:,1])

                mu_rv, std_rv = norm.fit(rv_pdf)
                p_rv = norm.pdf(x_rv, mu_rv, std_rv)
                ps_rv.append(p_rv)

                mu_fb, std_fb = norm.fit(fb_pdf)
                p_fb = norm.pdf(x_fb, mu_fb, std_fb)
                ps_fb.append(p_fb)

                n_rv, bins_rv, patches_rv = ax[0].hist(rv_pdf, bins=rvgrid, lw=3, histtype='step', normed=True, color=c, cumulative=cumulative)
                n_fb, bins_fb, patches_fb = ax[1].hist(fb_pdf, bins=fbgrid, lw=3, histtype='step', normed=True, color=c, cumulative=cumulative)

                #ax[0,0].plot(x_rv, p_rv, lw=1, color=c)
                #ax[0,1].plot(x_fg, p_fb, lw=1, color=c)
                #ax[0].plot(x_rv, p_rv, lw=1, color=c)
                #ax[1].plot(x_fb, p_fb, lw=1, color=c)

                nn_rv = np.log(n_rv)
                nn_rv[nn_rv == -np.inf] = 0.
                lps_rv.append(nn_rv)

                nn_fb = np.log(n_fb)
                nn_fb[nn_fb == -np.inf] = 0.
                lps_fb.append(nn_fb)

            lps_rv = np.asarray(lps_rv)
            lps_rv[lps_rv == 0.] = -100.
            lp_rv_pdf = np.sum(lps_rv, axis=0)
            #lp_rv_pdf /= 100.
            #lp_rv_pdf += np.max(lp_rv_pdf)*2

            lps_fb = np.asarray(lps_fb)
            lps_fb[lps_fb == 0.] = -100.
            lp_fb_pdf = np.sum(lps_fb, axis=0)
            #lp_fb_pdf /= 100.
            #lp_fb_pdf += np.max(lp_fb_pdf)*2

            #ax[1,0].hist(x_rv, bins=len(x_rv), weights=np.exp(lp_rv_pdf), lw=2, histtype='step', color='k', normed=True)
            #ax[1,1].hist(x_fb, bins=len(x_fb), weights=np.exp(lp_fb_pdf), lw=2, histtype='step', color='k', normed=True)



            ax[0].grid()
            ax[1].grid()
            ax[0].set_xlim(0.2, 10.2)
            ax[1].set_xlim(-0.05, 1.55)
            ax[0].set_ylim(0,2)

            #ax[0].set_yticklabels([])
            #ax[1].set_yticklabels([])

            #ax[0,0].set_ylabel('$\ln f$')
            ax[0].set_xlabel('$R_V$')
            #ax[1,0].set_ylabel('$\Sigma (\ln f)$')
            ax[1].set_xlabel('$f_{bump}$')

            #for a in ax:
            #    a[0].grid()
            #    a[1].grid()
            #    a[0].set_xlim(0.2, 10.2)
            #    a[1].set_xlim(-0.05, 1.55)

            #ax[0,0].set_xticklabels([])
            #ax[0,1].set_xticklabels([])

            #ax[0,0].set_ylabel('$\ln f$')
            #ax[1,0].set_xlabel('$R_V$')
            #ax[1,0].set_ylabel('$\Sigma (\ln f)$')
            #ax[1,1].set_xlabel('$f_{bump}$')

            if save:
                plt.savefig(data_loc + 'plots/reg_rv_fb_pdfs_1.pdf')
            else:
                plt.show()


def plot_walkers(infile, i):
    labels = ['$R_V$', r'$f_{\textrm{bump}}$']
    with h5py.File(infile, 'r') as hf:
        nregs = len(hf.keys())
        ndim = (hf.get(hf.keys()[i]))['sampler_flatchain'].shape[1]
        nwalkers = hf.get(hf.keys()[i])['sampler_chain'].shape[0]

        region = hf.get(hf.keys()[i])

        naxes = ndim + 1
        fig, ax = plt.subplots(naxes, 1, figsize=(naxes*1.5,naxes*2.8), sharex=True)

        pp = np.asarray([region['sampler_chain'][:,:, d] for d in range(ndim)])
        for aa in range(ndim):
            for w in range(nwalkers):
                ax[aa].plot(pp[aa, w, :])
        for w in range(nwalkers):
            ax[aa+1].plot(region['sampler_lnprob'][w,:])

        for i, ax in enumerate(fig.axes):
            if labels:
                labels.append('ln(prob)')
                ax.set_ylabel(labels[i])
            ax.tick_params(labelsize=13)
        fig.axes[-1].set_xlabel('Steps')
        plt.subplots_adjust(hspace=0.06, right=0.94, top=0.96, left=0.17,
                            bottom=0.08)
        plt.show()


save=True

data_loc = '/Users/alexialewis/research/PHAT/dustvar/'
group_labels = ['R_V', 'f_bump', 'sampler_chain', 'sampler_flatchain', 'sampler_lnprob']
#infile = os.path.join(data_loc, 'rv_fbump_1-50.hdf5')
infile = os.path.join(data_loc, 'all_runs.h5')

plot_pdfs([infile], save=save, regs=1, cumulative=False)

#plot_walkers(infile, 35)



