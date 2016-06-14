import numpy as np
import h5py
import matplotlib.pyplot as plt
import compile_data
import os
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pdb import set_trace
import copy
#import seaborn as sns

def gather_grid(infile):
    with h5py.File(infile, 'r') as hf:
        nregs = len(hf.keys())
        grid = np.asarray(np.zeros((nregs, 2)))

        grid[:,0] = np.asarray([hf.get(hf.keys()[i])['R_V'][1] for i in range(nregs)])
        grid[:,1] = np.asarray([hf.get(hf.keys()[i])['f_bump'][1] for i in range(nregs)])
    return grid


def adjust_cmap(cmap):
    colors = [(0.0,0.0,0.0)]
    colors.extend(cmap(np.linspace(0.14, 1, 99)))
    cmap = mcolors.ListedColormap(colors)
    cmap.set_bad('0.65')
    return cmap


def make_maps(rv, fb, rvlims=[0,10], fblims=[0,1.5], cmap1=plt.cm.inferno, cmap2=plt.cm.inferno, save=False, plotname=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5))
    plt.subplots_adjust(hspace=0.03, left=0.05, right=0.9, bottom=0.05, top=0.95)
    axlist = [ax1, ax2]
    for ax in axlist:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    rvmin, rvmax = rvlims[0], rvlims[1]
    fbmin, fbmax = fblims[0], fblims[1]
    dx1 = 0.05 * (rvmax - rvmin)
    dx2 = 0.05 * (fbmax - fbmin)
    cnorm1 = mcolors.Normalize(vmin=rvmin-dx1, vmax=rvmax+dx1)
    cnorm2 = mcolors.Normalize(vmin=fbmin-dx2, vmax=fbmax+dx2)

    cmap1.set_bad('0.65')
    cmap2.set_bad('0.65')
    if np.nanmin(rv) == -99.:
        cmap1 = adjust_cmap(cmap1)
    if np.nanmin(fb) == -99.:
        cmap2 = adjust_cmap(cmap2)

    im1 = ax1.imshow(rv[::-1].T, cmap=cmap1, norm=cnorm1)
    im2 = ax2.imshow(fb[::-1].T, cmap=cmap2, norm=cnorm2)

    ticks1 = np.linspace(rvmin, rvmax, 6)
    ticks2 = np.linspace(fbmin, fbmax, 6)

    pos1 = ax1.get_position()
    pos2 = ax2.get_position()

    divider1 = make_axes_locatable(ax1)
    cbax1 = divider1.append_axes('right', size="3%", pad=0.05)
    divider2 = make_axes_locatable(ax2)
    cbax2 = divider2.append_axes('right', size="3%", pad=0.05)

    ims = [im1, im2]
    cbax = [cbax1, cbax2]
    ticks = [ticks1, ticks2]
    labels = [r'$\Delta R_V$', r'$\Delta f_{bump}$']

    for i in range(len(axlist)):
        cb = fig.colorbar(ims[i], cax=cbax[i], orientation='vertical', ticks=ticks[i], drawedges=False)
        #cb.ax.xaxis.set_ticks_position('right')
        cb.ax.tick_params(labelsize=12)
        cb.set_label(labels[i], size=15, labelpad=5)

    if save:
        plt.savefig(plotname)
    else:
        plt.show()


def map1(rv, fb, other, data_loc='/Users/alexialewis/research/PHAT/dustvar', cmap1=plt.cm.Blues, cmap2=plt.cm.Reds, save=False):
    sfrsel = other['sfr100'] < 1e-5
    new_rv, new_fb = copy.copy(rv), copy.copy(fb)
    #new_rv[sfrsel] = -99.
    #new_fb[sfrsel] = -99
    if save:
        plotname = os.path.join(data_loc,'plots/rv_fbump_maps_nocut.pdf')
    else:
        plotname = None
    make_maps(new_rv, new_fb, rvlims=[2, 5], fblims=[0.4, 1.3], cmap1=cmap1, cmap2=cmap2, save=save, plotname=plotname)


def map2(rv, fb, other, data_loc='/Users/alexialewis/research/PHAT/dustvar', cmap1=plt.cm.RdBu, cmap2=plt.cm.RdBu, save=False):
    #m31_rv = 2.94
    #m31_fb = 0.68
    m31_rv = 3.01   #high sfrs
    m31_fb = 0.63
    rv_diff = rv - m31_rv
    fb_diff = fb - m31_fb
    sfrsel = other['sfr100'] < 1e-5
    rv_diff[sfrsel] = -99.
    fb_diff[sfrsel] = -99
    if save:
        plotname = os.path.join(data_loc, 'plots/rv_fbump_diffmaps_sfrcut_1e-5_new.pdf')
    else:
        plotname = None
    make_maps(rv_diff, fb_diff, rvlims=[-1.0, 1.0], fblims=[-0.3, 0.3], cmap1=cmap1, cmap2=cmap2, save=save, plotname=plotname)


def sfr_rv_fbump(rv, fb, other, data_loc='/Users/alexialewis/research/PHAT/dustvar', cmap=plt.cm.inferno):
    sns.reset_orig()
    from dustvar_paper_figs import make_2dhist_func
    sfr = other['sfr100'][np.isfinite(other['sfr100'])]
    x = rv[np.isfinite(rv)].flatten()
    y = np.log10(sfr.flatten())
    z = fb[np.isfinite(fb)].flatten()
    xlim = [0, 10]
    ylim = [-10, -3]
    zlim = [0.2, 1.2]
    zticks = np.linspace(zlim[0], zlim[1], 6)

    bins=75
    func = 'median'
    cnorm = mcolors.Normalize(vmin=zlim[0], vmax=zlim[-1])
    fmt = None
    xlabel = '$R_V$'
    ylabel = '$\log$ SFR \Big[M$_\odot$ yr$^{-1}$\Big]'
    zlabel = r'$f_{\rm bump}$'
    extend = 'neither'

    cmap = plt.cm.inferno

    hist_kwargs = {'func': func, 'bins': bins, 'xlim': xlim,
                   'ylim': ylim, 'cnorm': cnorm, 'cmap': cmap}

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    im, cnorm, bindata = make_2dhist_func(ax, x, y, z, **hist_kwargs)

    divider = make_axes_locatable(ax)
    cbax = divider.append_axes('right', size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cbax, norm=cnorm, orientation='vertical', ticks=zticks, extend='both')
    #cb.ax.xaxis.set_ticks_position('right')
    cb.ax.tick_params(labelsize=14)
    cb.set_label(zlabel, size=18, labelpad=5)
    cb.set_clim(vmin=zlim[0], vmax=zlim[1])

    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=16)

    plotname = os.path.join(data_loc, 'plots/sfr_rv_fbump.pdf')
    #plt.savefig(plotname)
    plt.show()


def main():
    data_loc1 = '/Users/alexialewis/research/PHAT/dustvar'

    sampler_file = os.path.join(data_loc1, 'all_runs.h5')

    # get median R_V and f_bump values of each pixel
    grid = gather_grid(sampler_file)

    # gather the CMD and flux data
    fuvdata, nuvdata, otherdata = compile_data.gather_map_data()
    sfr100 = otherdata['sfr100']

    # save the shape for future use
    shape = sfr100.shape

    # flatten the data
    sfr100flat = sfr100.flatten()

    # get the location of the finite values
    sel = np.isfinite(sfr100flat)

    rv_array = np.zeros(shape).flatten()
    fb_array = np.zeros(shape).flatten()

    rv_array[sel] = grid[:,0]
    fb_array[sel] = grid[:,1]
    rv_array[~sel] = np.nan
    fb_array[~sel] = np.nan

    rv = rv_array.reshape(shape)
    fb = fb_array.reshape(shape)

    #map1(rv, fb, otherdata, data_loc=data_loc1)
    #map2(rv, fb, otherdata, data_loc=data_loc1)
    #sfr_rv_fbump(rv, fb, otherdata, data_loc=data_loc1)
    return rv, fb


if __name__ == '__main__':
    rv, fb = main()

