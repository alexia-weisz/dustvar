import h5py
import numpy as np
import matplotlib.pyplot as plt



def write_to_screen(infile):
    with h5py.File(infile, 'r') as hf:
        print 'mu_rv: ', hf.get('R_V')['mu'][1],np.diff(hf.get('R_V')['mu'][:])
        print 'sigma_rv: ', hf.get('R_V')['sigma'][1], np.diff(hf.get('R_V')['sigma'][:])
        print 'mu_fb: ', hf.get('f_bump')['mu'][1], np.diff(hf.get('f_bump')['mu'][:])
        print 'sigma_fb: ', hf.get('f_bump')['sigma'][1], np.diff(hf.get('f_bump')['sigma'][:])


def plot_triangle(infile):
    import corner
    plotloc = '/Users/alexialewis/research/PHAT/dustvar/plots/'
    with h5py.File(infile, 'r') as hf:
        corner.ScalarFormatter(useOffset=False)
        sampler_rv = hf.get('R_V')['sampler_flatchain']
        sampler_fb = hf.get('f_bump')['sampler_flatchain']
        labels_rv = ['$\mu_{R_V}$', '$\sigma_{R_V}$']
        labels_fb = ['$\mu_{f_{bump}}$', '$\sigma_{f_{bump}}$']

        fig1 = corner.corner(sampler_rv,labels=labels_rv)#,range=lim)
        plotname1 = plotloc + 'triangle_rv_sfrgt-5.pdf'
        plt.savefig(plotname1)
        fig2 = corner.corner(sampler_fb, labels=labels_fb)
        plotname2 = plotloc + 'triangle_fb_sfrgt-5.pdf'
        plt.savefig(plotname2)
