import numpy as np
import h5py
import matplotlib.pyplot as plt
import compile_data
import os
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pdb import set_trace
import copy
import seaborn as sns
from scipy.stats import binned_statistic_2d

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


def make_2dhist_func(ax, x, y, z, func='median', bins=75, xlim=None, ylim=None,   vmin=0, vmax=1.5, origin='lower', aspect='auto', cnorm=None, cmap=plt.cm.Blues, interpolation='nearest'):
    """
    Make a 2d histogram binned along x and y colored with values in z undergone with
    func.
    Similar to histogram2d except allows for more than number counts to be in the
    histogram.
    """
    kwargs = {'statistic': func, 'bins': bins}
    if xlim is not None and ylim is not None:
        kwargs['range'] = [xlim, ylim]

    h, xe, ye, binnum = binned_statistic_2d(x, y, z, **kwargs)

    if ax is None:
        return h, xe, ye, binnum

    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    if cnorm is None:
        cnorm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(h.T, cmap=cmap, interpolation=interpolation, norm=cnorm,
                   aspect=aspect, extent=extent, origin='lower')
    return im, cnorm, [h, xe, ye, binnum]

def make_maps(rv, fb, rvlims=[0,10], fblims=[0,1.5], cmap1=plt.cm.inferno, cmap2=plt.cm.inferno, save=False, plotname=None, labels=None):

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
    if labels is None:
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


def map1(rv, fb, other, data_loc='/Users/alexialewis/research/PHAT/dustvar', cmap1=plt.cm.Blues, cmap2=plt.cm.Reds, save=False, cut=True):
    sfrsel = other['sfr100'] < 1e-5
    new_rv, new_fb = copy.copy(rv), copy.copy(fb)
    if cut:
        new_rv[sfrsel] = -99.
        new_fb[sfrsel] = -99
    xlabel = r'$\langle R_V \rangle$'
    ylabel = r'$\langle f_{\rm bump} \rangle$'
    labels = [r'$\langle R_V \rangle$', r'$\langle f_{\rm bump} \rangle$']

    plotname = os.path.join(data_loc,'plots/rv_fbump_maps_nocut.pdf')
    if cut:
        plotname = os.path.join(data_loc,'plots/rv_fbump_maps_sfrcut_1e-5.pdf')
    print plotname
    make_maps(new_rv, new_fb, rvlims=[2.4, 4.2], fblims=[0.5, 1.1], cmap1=cmap1, cmap2=cmap2, save=save, plotname=plotname, labels=labels)


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

    sfr = other['sfr100'][np.isfinite(other['sfr100'])]
    av = other['avdav'][np.isfinite(other['avdav'])]
    x = rv[np.isfinite(rv)].flatten()
    y = fb[np.isfinite(fb)].flatten()
    z1 = np.log10(sfr.flatten())
    z2 = av
    xlim = [0, 6]
    ylim = [0.2, 1.2]
    zlim1 = [-7, -3.5]
    zlim2 = [0, 1.5]
    zticks1 = np.linspace(zlim1[0], zlim1[1], 6)
    zticks2 = np.linspace(zlim2[0], zlim2[1], 6)

    bins=30
    func = 'median'
    cnorm1 = mcolors.Normalize(vmin=zlim1[0], vmax=zlim1[-1])
    cnorm2 = mcolors.Normalize(vmin=zlim2[0], vmax=zlim2[-1])
    fmt = None
    xlabel = r'$\langle R_V \rangle$'
    ylabel = r'$\langle f_{\rm bump} \rangle$'
    zlabel1 = '$\log$ SFR \Big[M$_\odot$ yr$^{-1}$\Big]'
    zlabel2 = '$\widetilde{A_V}$'
    extend = 'neither'

    cmap = plt.cm.inferno

    hist_kwargs = {'func': func, 'bins': bins, 'xlim': xlim,
                   'ylim': ylim, 'cmap': cmap}
    hist_kwargs2 = {'func': 'count', 'bins': bins, 'xlim': xlim,
                   'ylim': ylim, 'cmap': cmap}

    sel = x < 6

    counts1, xbins1, ybins1, bn = make_2dhist_func(None, x[sel], y[sel], z1[sel], cnorm=cnorm1, **hist_kwargs2)
    counts2, xbins2, ybins2, bn = make_2dhist_func(None, x[sel], y[sel], z2[sel], cnorm=cnorm2, **hist_kwargs2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,4), sharex=True, sharey=True)

    im1, cnorm1, bindata1 = make_2dhist_func(ax1, x[sel], y[sel], z1[sel], cnorm=cnorm1, **hist_kwargs)
    im2, cnorm2, bindata2 = make_2dhist_func(ax2, x[sel], y[sel], z2[sel], cnorm=cnorm2, **hist_kwargs)

    levels = [5, 15, 30]
    lw = 1
    ls = 'solid'
    color = '#00D8FF'
    ax1.contour(counts1.T, levels, linewidths=lw, colors=color, linestyles=ls, extent=[xbins1.min(),xbins1.max(),ybins1.min(),ybins1.max()])
    ax2.contour(counts2.T, levels, linewidths=lw, colors=color, linestyles=ls, extent=[xbins2.min(),xbins2.max(),ybins2.min(),ybins2.max()])

    axlist = [ax1, ax2]
    imlist = [im1, im2]
    labellist = [zlabel1, zlabel2]
    cnormlist = [cnorm1, cnorm2]
    ticklist = [zticks1, zticks2]
    extendlist = ['both', 'max']
    zlimlist = [zlim1, zlim2]
    for i, ax in enumerate(axlist):
        divider = make_axes_locatable(ax)
        cbax = divider.append_axes('top', size='5%', pad=0.05)
        cb = fig.colorbar(imlist[i], cax=cbax, norm=cnormlist[i], orientation='horizontal', ticks=ticklist[i], extend=extendlist[i])
        cbax.xaxis.set_ticks_position('top')
        cbax.xaxis.set_label_position('top')
        cb.ax.tick_params(labelsize=14, pad=5)
        cb.set_label(labellist[i], size=18, labelpad=5)
        cb.set_clim(vmin=zlimlist[i][0], vmax=zlimlist[i][1])
        ax.grid()
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlabel(xlabel)

    ax1.set_ylabel(ylabel)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    plt.subplots_adjust(wspace=0.1, left=0.1, right=0.9)
    plotname = os.path.join(data_loc, 'plots/sfr_av_rv_fbump_contours.pdf')
    #plt.savefig(plotname)
    plt.show()


def main():
    data_loc1 = '/Users/alexialewis/research/PHAT/dustvar'

    sampler_file = os.path.join(data_loc1, 'all_runs.h5')

    # get median R_V and f_bump values of each pixel
    #grid = gather_grid(sampler_file)
    grid = np.loadtxt(os.path.join(data_loc1, 'median_rv_fbump_per_reg.dat'))

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

    map1(rv, fb, otherdata, data_loc=data_loc1)
    #map2(rv, fb, otherdata, data_loc=data_loc1)
    #sfr_rv_fbump(rv, fb, otherdata, data_loc=data_loc1)
    return rv, fb


if __name__ == '__main__':
    rv, fb = main()

