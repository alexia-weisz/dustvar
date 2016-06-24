import numpy as np
import matplotlib.pyplot as plt
from sedpy import attenuation, observate
from scipy.stats import binned_statistic_2d
import matplotlib.colors as mcolors
import fsps
import compile_data
import os
from matplotlib.ticker import ScalarFormatter, LogFormatter
from pdb import set_trace

sps = fsps.StellarPopulation()
sps.params['sfh'] = 4
sps.params['const'] = 1.0
sps.params['imf_type'] = 2
wave, s = sps.get_spectrum(tage=1.0, peraa=True)

M31_DM = 24.47
ATT = attenuation.conroy

filters = ['galex_fuv', 'galex_nuv']
filters = observate.load_filters(filters)


def ext_func(rv, av=1.0, f_bump=1., att=attenuation.conroy):
    """
    Given an R_V and f_bump value, returns the flux ratio or delta color from a specific attenuation curve.

    Parameters
    ----------
    rv : float ; an R_V value
    av : float ; A_V for the given region
    f_bump : float, optional; strength of the 2175 \AA bump in fraction of MW bump strength
    att : sedpy.attenuation funcion, optional; attenuation curve to use. Default: attenuation.conroy
    """
    tau_v = av / 1.086
    tau_lambda = att(wave, R_v=rv, f_bump=f_bump, tau_v=tau_v)
    f2 = s * np.exp(-tau_lambda)
    mags_red = observate.getSED(wave, f2, filters)
    mags = observate.getSED(wave, s, filters)

    fluxes_red = [3631*10**(-0.4 * (mag + M31_DM)) for mag in mags_red]
    fluxes = [3631 * 10**(-0.4 * (mag + M31_DM)) for mag in mags]
    val_fuv = fluxes_red[0] / fluxes[0]
    val_nuv = fluxes_red[1] / fluxes[1]
    color = (mags_red[0] - mags_red[1]) - (mags[0] - mags[1])

    return val_fuv, val_nuv, color


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
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    if cnorm is None:
        cnorm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(h.T, cmap=cmap, interpolation=interpolation, norm=cnorm,
                   aspect=aspect, extent=extent, origin='lower')
    return im, cnorm, [h, xe, ye, binnum]



def create_cb(fig, axes_list, ims, ticks, cnorms=None, fmt=None, label='', labsize=14,pad=None,cbw=0.03,orientation='horizontal',extend='neither', cbarticksize=12):
    """
    Create a colorbar for an image plot (2d histogram or such). Can specify the
    ticks and orientation. This will add a new axis for the colorbar.
    """
    for i, ax in enumerate(axes_list):
        pos = ax.get_position()
        if orientation == 'horizontal':
            cax = fig.add_axes([pos.x0, pos.y1 + 0.01, pos.x1-pos.x0, cbw])
        else:
            cax = fig.add_axes([pos.x1 +0.02, pos.y0, cbw, pos.y1-pos.y0])
        if fmt == 'log':
            cform = LogFormatter(base=10, labelOnlyBase=False)
        else:
            cform = ScalarFormatter(useOffset=False)
        cb_kwargs = {'cax':cax, 'orientation': orientation, 'format':cform}
        if ticks is not None:
            cb_kwargs['ticks'] = ticks[i]
        if cnorms is not None:
            cb_kwargs['norm']=cnorms[i]

        cb = fig.colorbar(ims[i], cax=cax, orientation=orientation,
                          norm=cnorms[i], format=cform, ticks=ticks[i],
                          extend=extend)
        #cb = fig.colorbar(ims[i])#, **cb_kwargs)
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=cbarticksize)
        cb.set_label(label[i], size=labsize, labelpad=pad)


def theoretical_predictions():
    rvgrid = np.linspace(0.05, 6.0, 50)
    fbgrid = np.linspace(0.0, 3.0, 50)

    bins = min(rvgrid.shape[0], fbgrid.shape[0])

    rvg, fbg = np.meshgrid(rvgrid, fbgrid, indexing='ij')

    ebv = np.zeros(rvg.shape)
    fuvf = np.zeros(rvg.shape)
    nuvf = np.zeros(rvg.shape)
    for i in range(rvg.shape[0]):
        for j in range(rvg.shape[1]):
            rv, fb = rvg[i,j], fbg[i,j]
            model = ext_func(rv, av=1.4, f_bump=fb)
            ebv[i,j] = model[-1]
            fuvf[i,j] = model[0]
            nuvf[i,j] = model[1]

    ebv_cmin, ebv_cmax = np.nanmin(ebv), 3#np.nanmax(ebv)
    ebv_cmap = plt.cm.inferno
    ebv_cmap.set_under('white')
    ebv[ebv > 10] = -999

    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, ax3 = plt.subplots(1, 1)
    #im1, cnorm1, extra1 = make_2dhist_func(ax1, rvg.flatten(), fbg.flatten(), fuvf.flatten(), func='median')


    im3, cnorm3, extra3 = make_2dhist_func(ax3, rvg.flatten(), fbg.flatten(), ebv.flatten(), func='median', cmap=ebv_cmap, bins=bins, vmin=ebv_cmin, vmax=ebv_cmax)
    cb = plt.colorbar(im3, extend='max')
    cb.set_label('E(FUV - NUV)', size=14)
    ax3.set_xlabel('$R_V$', fontsize=16)
    ax3.set_ylabel('$f_{bump}$', size=16)

    plt.show()


def get_grid_data(first=False):
    if first:
        import h5py
        hf_file = os.path.join(_WORK_DIR, 'all_runs.h5')
        hf = h5py.File(hf_file, 'r')

        nregs = len(hf.keys())
        reg_range = range(nregs)
        nsamples = hf.get(hf.keys()[0])['sampler_chain'].shape[1]
        grid = np.asarray(np.zeros((nregs, 2)))
        for i, reg in enumerate(reg_range):
            group = hf.get(hf.keys()[reg])
            grid[i,0] = np.median(np.asarray(group['sampler_flatchain'][:,0]))
            grid[i,1] = np.median(np.asarray(group['sampler_flatchain'][:,1]))
        hf.close()
        np.savetxt(gridfile, grid)
    else:
        grid = np.loadtxt(gridfile)

    return grid


def grid_to_arrays(grid, otherdata):
    sfr100 = otherdata['sfr100']
    shape = sfr100.shape
    sfr100flat = sfr100.flatten()
    sel = np.isfinite(sfr100flat)
    sel2 = np.isfinite(sfr100)

    #create empty arrays
    rv_array = np.zeros(shape).flatten()
    sig_rv_array = np.zeros(shape).flatten()
    fb_array = np.zeros(shape).flatten()
    sig_fb_array = np.zeros(shape).flatten()

    # fill arrays and reshape
    rv_array[sel] = grid[:,0]
    fb_array[sel] = grid[:,1]
    rv_array[~sel] = np.nan
    fb_array[~sel] = np.nan
    rv = rv_array.reshape(shape)
    fb = fb_array.reshape(shape)

    return rv, fb



if __name__ == '__main__':

    res = '90'
    dust_curve='cardelli'
    sfh = 'full_sfh'

    if os.environ['PATH'][1:6] == 'astro':
        _TOP_DIR = '/astro/store/phat/arlewis/'
    else:
        _TOP_DIR = '/Users/alexialewis/research/PHAT/'

    _DATA_DIR = os.path.join(_TOP_DIR, 'maps', 'analysis')
    _WORK_DIR = os.path.join(_TOP_DIR, 'dustvar')
    _PLOT_DIR = os.path.join(_WORK_DIR, 'plots')

    gridfile = os.path.join(_WORK_DIR, 'median_rv_fbump_per_reg.dat')


    kwargs = {'save': False}

    fuvdata, nuvdata, otherdata = compile_data.gather_map_data(res, dust_curve, sfh=sfh)

    first = False

    grid = get_grid_data(first=False)

    rv, fb = grid_to_arrays(grid, otherdata)

    fuvf = np.log10(fuvdata['fluxobs'] / fuvdata['fluxmodred'])
    nuvf = np.log10(nuvdata['fluxobs'] / nuvdata['fluxmodred'])
    ebv = (fuvdata['magmodred'] - nuvdata['magmodred'] - (fuvdata['magobs'] - nuvdata['magobs']))

    ebv_cmin, ebv_cmax = -1, 1#-0.05, 0.25#np.nanmin(ebv), 0.5#np.nanmax(ebv)
    ebv_cmap = plt.cm.coolwarm
    ebvticks = np.linspace(ebv_cmin, ebv_cmax, 6)

    flux_cmin, flux_cmax = -0.5, 0.5
    flux_cmap = plt.cm.coolwarm
    fluxticks = np.linspace(flux_cmin, flux_cmax, 7)

    bins = 50

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.03, top=0.85)

    im1, cnorm1, extra1 = make_2dhist_func(ax1, rv[np.isfinite(rv)], fb[np.isfinite(fb)], fuvf[np.isfinite(fuvf)], func='median', cmap=flux_cmap, bins=bins, vmin=flux_cmin, vmax=flux_cmax)

    im2, cnorm2, extra2 = make_2dhist_func(ax2, rv[np.isfinite(rv)], fb[np.isfinite(fb)], nuvf[np.isfinite(nuvf)], func='median', cmap=flux_cmap, bins=bins, vmin=flux_cmin, vmax=flux_cmax)

    im3, cnorm3, extra3 = make_2dhist_func(ax3, rv[np.isfinite(rv)], fb[np.isfinite(fb)], ebv[np.isfinite(ebv)], func='median', cmap=ebv_cmap, bins=bins, vmin=ebv_cmin, vmax=ebv_cmax)

    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()

    cbax1 = fig.add_axes([pos1.x0, pos1.y1+0.02, pos2.x1 - pos1.x0, 0.04])
    cbax2 = fig.add_axes([pos3.x0, pos3.y1+0.02, pos3.x1 - pos3.x0, 0.04])
    cbticks = 10

    cb1 = fig.colorbar(im1, cax=cbax1, orientation='horizontal', norm=cnorm1, extend='both', ticks=fluxticks)
    cb1.ax.tick_params(labelsize=10, pad=-25)
    cb1.set_label(r'$f^{\rm obs}/f^{\rm mod,red}$', size=12, labelpad=-40)

    cb2 = fig.colorbar(im3, cax=cbax2, orientation='horizontal', norm=cnorm3, extend='max', ticks=ebvticks)
    cb2.ax.tick_params(labelsize=10, pad=-25)
    cb2.set_label(r'$(m_{\rm \small FUV} - m_{\rm \small NUV})_{\rm obs} - (m_{\rm \small FUV} - m_{\rm \small NUV})_{\rm mod,red}$', size=12, labelpad=-40)

    ax2.set_xlabel(r'$R_V$', size=18)
    ax1.set_ylabel(r'$f_{\rm bump}$', size=18)

    ax1.grid()
    ax2.grid()
    ax3.grid()

    xlim, ylim = ax1.get_xlim(), ax1.get_ylim()

    n = 0.05
    rve1, fbe1 = (extra1[1][1:] + extra1[1][:-1]) / 2., (extra1[2][1:] + extra1[2][:-1]) / 2.
    rve2, fbe2 = (extra2[1][1:] + extra2[1][:-1]) / 2., (extra2[2][1:] + extra2[2][:-1]) / 2.
    rve3, fbe3 = (extra3[1][1:] + extra3[1][:-1]) / 2., (extra3[2][1:] + extra3[2][:-1]) / 2.
    rvg1, fbg1 = np.meshgrid(rve1, fbe1, indexing='ij')
    rvg2, fbg2 = np.meshgrid(rve2, fbe2, indexing='ij')
    rvg3, fbg3 = np.meshgrid(rve3, fbe3, indexing='ij')
    selfuv = np.where((extra1[0] > -n) & (extra1[0] < n) & (rvg1 < 5.))
    selnuv = np.where((extra2[0] > -n) & (extra2[0] < n) & (rvg2 < 5.))
    selcol = np.where((extra3[0] > -n) & (extra3[0] < n))# & (rvg3 < 5.))

    x1, y1 = rve1[selfuv[0]], fbe1[selfuv[1]]
    x2, y2 = rve2[selnuv[0]], fbe2[selnuv[1]]
    x3, y3 = rve3[selcol[0]], fbe3[selcol[1]]

    from scipy.optimize import curve_fit

    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A*x + B


    A1 = np.vstack([x1, np.ones(len(x1))]).T
    m1, b1 = np.linalg.lstsq(A1, y1)[0]
    A2 = np.vstack([x2, np.ones(len(x2))]).T
    m2, b2 = np.linalg.lstsq(A2, y2)[0]
    A3 = np.vstack([x3, np.ones(len(x3))]).T
    m3, b3 = np.linalg.lstsq(A3, y3)[0]

    xx1 = np.linspace(np.min(x1), np.max(x1), 100)
    xx2 = np.linspace(np.min(x2), np.max(x2), 100)
    xx = np.linspace(0, 10, 100)
    yy1 = m1 * xx + b1
    yy2 = m2 * xx + b2
    yy3 = m3 * xx + b3

    from shapely.geometry import LineString
    line1 = LineString([(xx[0], yy1[0]), (xx[-1], yy1[-1])])
    line2 = LineString([(xx[0], yy2[0]), (xx[-1], yy2[-1])])
    line3 = LineString([(xx[0], yy3[0]), (xx[-1], yy3[-1])])
    intersect1 = line1.intersection(line2)
    intersect2 = line1.intersection(line3)
    intersect3 = line3.intersection(line2)
    print intersect1, intersect2, intersect3

    ax1.plot(xx, yy1, 'k-', lw=3)
    ax1.plot(xx, yy2, 'k--', lw=2)
    ax1.plot(xx, yy3, 'k:', lw=2)
    ax2.plot(xx, yy2, 'k-', lw=3)
    ax2.plot(xx, yy1, 'k--', lw=2)
    ax2.plot(xx, yy3, 'k:', lw=2)
    ax3.plot(xx, yy3, 'k-', lw=3)
    ax3.plot(xx, yy1, 'k--', lw=2)
    ax3.plot(xx, yy2, 'k:', lw=2)

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    plotname = os.path.join(_PLOT_DIR, 'rv_fb_data_expectations.pdf')
    if kwargs['save']:
        plt.savefig(plotname)
    else:
        plt.show()



    #theoretical_predictions()
