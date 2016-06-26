import numpy as np
import os
import compile_data
import astrogrid
import match_utils
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams

from scipy.stats import binned_statistic_2d

from astropy.coordinates import Distance
import astropy.units as u
import astropy.constants as const
#import plot_map_fig1 as plot_fig1

from sys import exit
from pdb import set_trace


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--res', default='90')
    parser.add_argument('--dust_curve', default='cardelli')
    parser.add_argument('--density', action='store_true', help='Make traditional 2d histograms based on counts per bin. If false, a third quantity is used.')
    parser.add_argument('--outline', action='store_true', help='Outline selected regions on the plot.')
    parser.add_argument('--sfhs', action='store_true', help='Plot small SFHs from outlined selected regions.')
    parser.add_argument('--red', action='store_true', help='Use reddened rather than intrinsic modeled flux.')
    parser.add_argument('--sfh', default='full_sfh', choices=['full_sfh',
                        'sfh1000', 'sfh500', 'sfh200', 'sfh100'])
    parser.add_argument('--ybar_outline', action='store_true', help='Plot horizontal line showing y-axis values over which sfhs are valid.')
    return parser.parse_args()

def set_colormaps():
    cmap = plt.cm.gist_heat_r
    cmap2, cmap3, cmap4 = plt.cm.Blues_r, plt.cm.Purples_r, plt.cm.Greens
    cmap_resid = plt.cm.coolwarm

    for cm in [cmap, cmap2, cmap3, cmap4, cmap_resid]:
        cm.set_bad('0.65')
        #cm.set_under('0.75')
    return cmap, cmap2, cmap3, cmap4, cmap_resid

def get_seaborn_colors(n_colors):
    colors = ['dark pink', 'reddish pink', 'coral', 'orange', 'dark periwinkle']
    return sns.xkcd_palette(colors)
    #return sns.color_palette('colorblind', n_colors=n_colors)

def create_cb(fig, axes_list, ims, ticks, cnorms=None, fmt=None, label='', labsize=14, pad=None, cbw=0.03, orientation='horizontal', extend='neither'):
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
        cb.ax.tick_params(labelsize=CBARTICKSIZE)
        cb.set_label(label[i], size=labsize, labelpad=pad)

def make_2dhist(ax, x, y, bins, xlim=None, ylim=None, vmin=1, vmax=500, origin='lower', aspect='auto', cmap=plt.cm.Blues, interpolation='nearest'):
    """
    Make a 2d histogram of number counts binned along x and y.
    """
    hist_kwargs = {'bins': bins}
    if xlim is not None and ylim is not None:
        hist_kwargs['range'] = [xlim, ylim]
    h, xe, ye = np.histogram2d(x, y, **hist_kwargs)
    h[h < vmin] = -999
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    cnorm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(h.T, cmap=cmap, interpolation=interpolation, extent=extent,
                   aspect=aspect, norm=cnorm, origin=origin)
    return im, cnorm, h, xe, ye

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

def running_median(X, Y, ax, total_bins=9, min=-16.3, max=-13.7, color='r', flux_ratio=True, check=False):
    bins = np.linspace(min, max, total_bins)
    delta = bins[1] - bins[0]
    #print delta
    idx = np.digitize(X, bins)
    run_med = [np.nanmedian(Y[idx==k]) for k in range(1, total_bins)]
    run_std = [np.nanstd(Y[idx==k]) for k in range(1, total_bins)]
    run_count = [len(Y[idx==k]) for k in range(1, total_bins)]
    for pp in [1, 2, 3]:
        run_percent = [len(Y[idx==k][(Y[idx==k] < pp*run_std[k-1]) & (Y[idx==k] > - pp*run_std[k-1])]) / float(run_count[k-1]) if run_count[k-1] > 0 else -999 for k in range(1, total_bins)]
        if flux_ratio:
            print '   percent within ' + str(pp) + ' sigma: '
            print run_percent
            print run_count
    if check: set_trace()
    #set_trace()
    #print ' '
    xx = [bins[j] + delta/2. for j in range(total_bins-1)]
    yy = run_med
    ax.errorbar(xx, yy, yerr=run_std, marker='o', ls='--', color=color, mec=color, lw=3, elinewidth=2, zorder=100)


def create_bins(X, Y, total_bins=8, min=None, max=None, color='r'):
    #X = X[np.isfinite(X)]
    #y = Y[np.isfinite(Y)]
    #min, max = -16.8, -13.1

    if min is None:
        min = np.nanmin(X)
    if max is None:
        max = np.nanmax(X)
    bins = np.linspace(min, max, total_bins)
    delta = bins[1] - bins[0]
    idx = np.digitize(X, bins)
    return idx, bins, delta

def sum_in_quad(val):
    val = val[np.isfinite(val)]
    unc = np.sqrt(np.sum(val**2)) / np.sqrt(len(val))
    return unc

def draw_grid(ax=None):
    ax.grid(color='k', linestyle='-', linewidth=0.5, alpha=0.1, zorder=500)

def get_paths():
    from matplotlib.path import Path
    regfile = os.path.join(_WORK_DIR, 'ring_sels_im.reg')
    f = open(regfile)
    lines = f.readlines()
    f.close()
    reg_paths = []
    polys = lines[3:]
    for p in polys:
        pp = p.strip('polygon()\n').split(',')
        ppx = np.array([float(ppp) for ppp in pp[0::2]])
        ppy = np.array([float(ppp) for ppp in pp[1::2]])
        path = Path(zip(ppx, ppy))
        reg_paths.append(path)
    return reg_paths

def get_numbers_obscured_sf(fuvdata, flux_fuv_24):

    x, y = np.arange(fuvdata['fluxobs'].shape[0]), np.arange(fuvdata['fluxobs'].shape[1])
    X, Y = np.meshgrid(x, y)
    #xmids = (X[:, 1:] - 0.5).flatten()
    #ymids = (Y[1:, :] - 0.5).flatten()
    #pixels = np.array(zip(ymids,xmids))
    xx, yy = X.flatten(), Y.flatten()
    pixels = np.array(zip(yy, xx))

    reg_paths = get_paths()

    ring10pathsel = reg_paths[0].contains_points(pixels)
    ring15pathsel = reg_paths[1].contains_points(pixels)
    ring05pathsel = reg_paths[2].contains_points(pixels)
    other = (~ring05pathsel & ~ring10pathsel & ~ring15pathsel)
    sels = [ring05pathsel, ring10pathsel, ring15pathsel, other]

    def flux_sums(data, paths, func=np.nanmedian):
        data_dict = {}
        data_dict['ring05'] = func(data.flatten()[sels[0]])
        data_dict['ring10'] = func(data.flatten()[sels[1]])
        data_dict['ring15'] = func(data.flatten()[sels[2]])
        data_dict['other'] = func(data.flatten()[sels[3]])
        return data_dict

    fuvobsflux = flux_sums(fuvdata['fluxobs'], sels)
    fuv24flux = flux_sums(flux_fuv_24, sels)
    fuvmodredflux = flux_sums(fuvdata['fluxmodred'], sels)
    fuvmodintflux = flux_sums(fuvdata['fluxmodint'], sels)

    ## compare the modeled data to itself
    print fuvmodintflux['ring05'] / fuvmodredflux['ring05'], fuvmodintflux['ring10'] / fuvmodredflux['ring10'], fuvmodintflux['ring15'] / fuvmodredflux['ring15'], fuvmodintflux['other'] / fuvmodredflux['other']
    ## compare the observed data to itself
    print fuv24flux['ring05'] / fuvobsflux['ring05'], fuv24flux['ring10'] / fuvobsflux['ring10'], fuv24flux['ring15'] / fuvobsflux['ring15'], fuv24flux['other'] / fuvobsflux['other']

    ## compare the modeled data to the observed data
    print fuvmodintflux['ring05'] / fuv24flux['ring05'], fuvmodintflux['ring10'] / fuv24flux['ring10'], fuvmodintflux['ring15'] / fuv24flux['ring15'], fuvmodintflux['other'] / fuv24flux['other']

    total_area_model_percent = 1. - np.nansum(fuvdata['fluxmodred'])/np.nansum(fuvdata['fluxmodint'])
    total_area_obs_percent = 1. - np.nansum(fuvdata['fluxobs'])/np.nansum(flux_fuv_24)

    print np.nansum(fuvdata['fluxmodint']) / np.nansum(fuvdata['fluxmodred'])
    print np.nansum(flux_fuv_24) / np.nansum(fuvdata['fluxobs'])
    print np.nansum(fuvdata['fluxmodint']) / np.nansum(flux_fuv_24)
    print np.nansum(fuvdata['fluxmodred']) / np.nansum(fuvdata['fluxobs'])
    print total_area_model_percent, total_area_obs_percent
    print np.nanmin(1. - fuvdata['fluxmodred']/fuvdata['fluxmodint']), np.nanmax(1. - fuvdata['fluxmodred']/fuvdata['fluxmodint'])
    print np.nanmin(1. - fuvdata['fluxobs'] / flux_fuv_24), np.nanmax(1. - fuvdata['fluxobs'] / flux_fuv_24)


class Dust(object):
    """
    Determine the FUV dust attenuation in order to correct FUV magnitudes and then convert to SFR.
    """
    def __init__(self, fuvdata, nuvdata, otherdata):
        self.d = Distance(distmod=24.47).to(u.cm).value  #distance to M31 in cm
        self.c = const.c.to('angstrom/s').value
        self.lambda_fuv = (1528 * u.angstrom).value
        self.lambda_24 = (24 * u.micron).to(u.angstrom).value
        self.nufuv = self.c / self.lambda_fuv
        self.nu24 = self.c / self.lambda_24
        self.lnu_fuv = fuvdata['fluxobs'] * (self.lambda_fuv**2 / self.c) * (4. * np.pi * self.d**2)
        self.lnu_24 = otherdata['mips24'] * (self.lambda_24**2 / self.c) * (4. * np.pi * self.d**2)
        self.nu_lnu_fuv = self.nufuv * self.lnu_fuv
        self.nu_lnu_24 = self.nu24 * self.lnu_24

    def from_tau(self, fuvdata):
        """
        a_lambda = 1.086 * tau_lambda
        """
        fuv_lambda = np.atleast_1d(1528) #angstrom
        dust_curve = attenuation.cardelli
        tau_lambda = dust_curve(fuv_lambda)
        a_fuv = 1.086 * tau_lambda
        return a_fuv

    def from_24micron_nu_lnu_ben(self):
        eta = 3.89
        a_fuv = 2.5 * np.log10(1. + eta * (self.nu_lnu_24 / self.nu_lnu_fuv))
        return a_fuv

    def from_24micron_hao(self):
        IRX = np.log10(self.nu_lnu_24 / (.12 * self.nu_lnu_fuv))
        a_fuv = 2.5 * np.log10(1. + .46 * 10**IRX)
        return a_fuv

    def from_color(self, fuvdata, nuvdata):
        a_fuv = 3.83 * (fuvdata['magobs'] - nuvdata['magobs']) - 0.022
        return a_fuv

    def from_sfh_modeled(self, fuvdata):
        """
        Want magnitude data here, not flux data
        a_lambda = modeled_reddened_mag - modeled_unreddened_mag
        """
        a_fuv = fuvdata['magmodred'] - fuvdata['magmodint']
        return a_fuv

class SFR(object):
    """
    Convert FUV to SFR.
    """
    def __init__(self, fuvdata, nuvdata, otherdata):
        self.dust = Dust(fuvdata, nuvdata, otherdata)
        self.afuv_24micron_ben = self.dust.from_24micron_nu_lnu_ben()
        self.afuv_24micron_hao = self.dust.from_24micron_hao()
        self.afuv_color = self.dust.from_color(fuvdata, nuvdata)
        self.afuv_sfh_modeled = self.dust.from_sfh_modeled(fuvdata)
        self.lambda_fuv = (1528 * u.angstrom).value
        self.d = Distance(distmod=24.47)
        self.lambda_mips = (24 * u.micron).to(u.angstrom).value
        self.nu_mips = (1.25e13 * u.Hz).value

    def correct_for_dust(self, m_fuv, m_dust):
        m_corr = m_fuv - m_dust
        flux_corr = astrogrid.flux.galex_mag2flux(m_corr, 'galex_fuv')
        return flux_corr

    def kennicutt(self, flux, is_corrected=False, ben=False, hao=False, color=False, modeled=False):
        """
        flux can be corrected for dust or not. Will correct for dust if needed, then convert to luminosity and compute sfr.

        Dust descriptions:
        -- set ben to True to use afuv_24micron_ben
        -- set hao to True to use afuv_24micron_hao
        -- set color to True to use afuv_color
        -- set modeled to True to use afuv_sfh_modeled

        flux units are erg s-1 cm-2 A-1
        """
        if not is_corrected:
            if ben: m_dust = self.afuv_24micron_ben
            if hao: m_dust = self.afuv_24micron_hao
            if color: m_dust = self.afuv_color
            if modeled: m_dust = self.afuv_sfh_modeled

            m_fuv = astrogrid.flux.galex_flux2mag(flux, 'galex_fuv')
            flux = self.correct_for_dust(m_fuv, m_dust)

        total_flux = flux * self.lambda_fuv #erg s-1 cm-2
        L_fuv = total_flux * (4. * np.pi * (self.d).to(u.cm).value**2) #erg s-1
        sfr = (10**-43.35 * L_fuv)
        return sfr

    def fuv_24micron(self, fuvdata, mipsdata):
        """
        Determine SFR from FUV + 24 micron data.
        FUV and 24 micron units are erg s-1 cm-2 A-1
        SFR prescription from Leroy et al. 2008
        SFR_tot = 0.68e-28 L_nu(FUV) + 2.14e-42 L(24 micron)
        0.68 is the 1.08 term from Salim et al 2007 divided by 1.59 to convert from Salpter to Kroupa
        and fro Hao et al. 2011 assumes Kroupa IMF
        SFR = 4.46e-44 * (L(FUV) + 3.89 L(24 micron)) in erg s-1
        """
        # first convert to luminosity units (erg s-1 Hz-1)
        c = const.c.to('angstrom/s').value
        fuv_lum_nu = (4 * np.pi * (self.d).to(u.cm).value**2) * (self.lambda_fuv**2 / c) * fuvdata
        mips_lum_nu = (4 * np.pi * (self.d).to(u.cm).value**2) * (self.lambda_mips**2 / c) * mipsdata
        fuv_lum = (4 * np.pi * (self.d).to(u.cm).value**2) * self.lambda_fuv * fuvdata
        mips_lum = (4 * np.pi * (self.d).to(u.cm).value**2) * self.lambda_mips * mipsdata
        sfr1 = 0.68e-28 * fuv_lum_nu + 2.14e-43 * mips_lum  #from Leroy
        #sfr2 = 4.46e-44 * (fuv_lum + 3.89 * mips_lum)  #from Hao
        sfr2 = 10**-43.35 * (fuv_lum + 3.89 * mips_lum)  #from Hao
        flux_fuv_24 = (fuv_lum + 3.89 * mips_lum) / (4 * np.pi * (self.d).to(u.cm).value**2) / self.lambda_fuv
        #flux_fuv_24b = sfr3 / 4.46e-44 / (4 * np.pi * (self.d).to(u.cm).value**2) / self.lambda_fuv
        return sfr1, sfr2, flux_fuv_24

    def fuv(self, fuvdata):
        """
        Determine SFR from FUV data.
        FUV units are erg s-1 cm-2 A-1
        SFR prescription from Leroy et al. 2008
        SFR = 0.68e-28 L(FUV) [erg s-1 Hz-1]
        """
        c = const.c.to('angstrom/s').value
        fuv_lum_nu = (4 * np.pi * (self.d).to(u.cm).value**2) * (self.lambda_fuv**2 / c) * fuvdata
        sfr = 0.68e-28 * fuv_lum_nu
        return sfr


def fig1(**kwargs):
    plot_fig1.main(plot_dir=_PLOT_DIR, data_dir=_WORK_DIR, save=kwargs['save'])

def fig2_3(uvdata, **kwargs):
    """
    Plot maps in the observed and modeled flux as well as residual significance between the two.

    """
    if np.array_equal(uvdata, fuvdata):#uvdata == fuvdata:
        #labels = [fuvfluxobslab, fuvfluxmodredlab, fuvfluxmodintlab]
        #labels = [fuvfluxobslab, fuvfluxmodredlab, fuvfluxobslab + '-' + fuvfluxmodredlab]
        labels = [fuvfluxobslab, fuvfluxmodredlab, r'$\frac{\small{'+fuvfluxobslab+'-'+fuvfluxmodredlab+'}}{\small{'+fuvfluxmodredlab+'}}$']
        plotname = os.path.join(_PLOT_DIR,'fuv_maps.'+plot_kwargs['format'])
    else:
        #labels = [nuvfluxobslab, nuvfluxmodredlab, nuvfluxobslab + '-' + nuvfluxmodredlab]
        labels = [nuvfluxobslab, nuvfluxmodredlab, r'$\frac{\small{'+nuvfluxobslab+'-'+nuvfluxmodredlab+'}}{\small{'+nuvfluxmodredlab+'}}$']
        plotname = os.path.join(_PLOT_DIR,'nuv_maps.'+plot_kwargs['format'])

    residuals = uvdata['fluxobs'] - uvdata['fluxmodred']
    #resid_sig = (uvdata['fluxmodred'] - uvdata['fluxobs']) / uvdata['fluxobs']
    resid_sig = residuals / uvdata['fluxmodred']

    n = uvdata['fluxobs']
    m = uvdata['fluxmodred']
    sig = np.sqrt(2*(m+n*(np.log(n/m)-1.0)))*np.sign(n-m)


    #set_trace()
    rcParams['axes.linewidth'] = 1
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6,6))

    ff = 1e-16
    cnorm1 = mcolors.Normalize(vmin=fluxmin/ff, vmax=fluxmax/ff)

    resid_vmin, resid_vmax = -5, 5
    cnorm2 = mcolors.Normalize(vmin=resid_vmin, vmax=resid_vmax)

    residsig_vmin, residsig_vmax = -3, 3
    cnorm3_residsig = mcolors.Normalize(vmin=residsig_vmin, vmax=residsig_vmax)
    cnorm3_sig = mcolors.Normalize(vmin=-0.25, vmax=0.25)

    im1 = ax1.imshow(uvdata['fluxobs']/ff, cmap=cmap, origin=origin,
                     interpolation=interpolation, norm=cnorm1)
    im2 = ax2.imshow(uvdata['fluxmodred']/ff, cmap=cmap, origin=origin,
                     interpolation=interpolation, norm=cnorm1)
    #im3 = ax3.imshow(residuals/ff, origin=origin, interpolation=interpolation,
    #                 cmap=cmap_resid, norm=cnorm2)
    #im3 = ax3.imshow(sig/1e-7, origin=origin, interpolation=interpolation,
    #                 cmap=cmap_resid, norm=cnorm3_sig)
    im3 = ax3.imshow(resid_sig, origin=origin, interpolation=interpolation,
                     cmap=cmap_resid, norm=cnorm3_residsig)

    plt.subplots_adjust(wspace=0.03, left=0.05, right=0.95, bottom=0.01, top=0.88)

    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()

    ticks1 = np.linspace(fluxmin/ff, fluxmax/ff, 8)
    ticks2 = np.arange(-5, 6, 2)
    ticks2_sig = np.linspace(-0.5, 0.5, 5)
    ticks2_residsig = np.linspace(residsig_vmin, residsig_vmax, 5)

    cbax1 = fig.add_axes([pos1.x0, pos1.y1-.01, pos2.x1 - pos1.x0, 0.03])
    cbax2 = fig.add_axes([pos3.x0, pos3.y1-.01, pos3.x1 - pos3.x0, 0.03])

    cb1 = fig.colorbar(im1, cax=cbax1, orientation='horizontal', extend='both')
    cb1.ax.xaxis.set_ticks_position('top')
    cb1.ax.tick_params(labelsize=CBARTICKSIZE)
    cb1.set_label(r'Flux $\times 10^{-16}$ [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]', size=12, labelpad=-45)

    cb2 = fig.colorbar(im3, cax=cbax2, orientation='horizontal', extend='both',
                       ticks=ticks2_residsig)
    cb2.ax.xaxis.set_ticks_position('top')
    cb2.ax.tick_params(labelsize=CBARTICKSIZE)
    #cb2.set_label(r'Residuals $\times 10^{-16}$', size=12, labelpad=-45)
    cb2.set_label(r'Residual Significance', size=12, labelpad=-42)

    #txtloc = [[0.04, 0.94]] * 2 + [[0.60, 0.02]]
    txtloc = [[0.80, 0.05]] * 3
    ptxtsize = [PLOTTEXTSIZE] * 2 + [PLOTTEXTSIZE-2]
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(txtloc[i][0], txtloc[i][1], labels[i], transform=ax.transAxes,
                fontsize=PLOTTEXTSIZE, ha='center', va='center')

    if kwargs['save']:
        plt.savefig(plotname, **plot_kwargs)
    else:
        plt.show()
    rcParams['axes.linewidth'] = 2

def fig4(fuvdata, nuvdata, ratio=True, two=True, single=None, **kwargs):
    """
    f^syn_red/f^obs vs f_obs.
    Ratio of modeled reddened flux to observed flux as a function of observed flux. Also plots running median.
    """
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    #plt.subplots_adjust(wspace=0.4, top=0.85, right=0.95)


    #fuvdata['fluxobs'] = astrogrid.flux.mag2flux(fuvdata['magobs'] - 0.2, 'galex_fuv')
    #nuvdata['fluxobs'] = astrogrid.flux.mag2flux(nuvdata['magobs'] - 0.2, 'galex_nuv')

    if two:
        if kwargs['vertical']:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,8))
            plt.subplots_adjust(hspace=0.2, top=0.95, right=0.85)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
            plt.subplots_adjust(wspace=0.4, top=0.85, right=0.95)
    else:
        fig = plt.figure(figsize=(5,5))
        ax1 = fig.add_subplot(111)

    x1 = np.log10(fuvdata['fluxobs'])
    x2 = np.log10(nuvdata['fluxobs'])
    #x2 = np.log10(nuvdata['fluxobs'])
    xlim = [-17.4, -12.2]
    fluxunits = r'  [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]'
    xlabel1 = r'$\log$ ' + fuvfluxobslab + fluxunits
    #xlabel2 = r'$\log ' + nuvfluxobslab
    xlabel2 = r'$\log$ ' + fuvfluxobslab + fluxunits

    if single is None:
        xlabel2 = r'$\log$ ' + nuvfluxobslab + fluxunits

    if single == fuvdata:
        cmapa, cmapb = cmap2, cmap2
    if single == nuvdata:
        cmapa, cmapb = cmap3, cmap3
    else:
        cmapa, cmapb = cmap2, cmap3

    cmapa.set_bad('white')
    cmapb.set_bad('white')
    #set_trace()
    if ratio:
        if single is None:
            y1 = np.log10(fuvdata['fluxmodred'] / fuvdata['fluxobs'])
            y2 = np.log10(nuvdata['fluxmodred'] / nuvdata['fluxobs'])
            ylabel1 = (r'$\log \Bigg(' + fuvfluxmodredlab + '/' +
                       fuvfluxobslab + '\Bigg)$')
            ylabel2 = (r'$\log \Bigg(' + nuvfluxmodredlab + '/' +
                       nuvfluxobslab + '\Bigg)$')
        else:
            y1 = np.log10(single['fluxmodint'] / single['fluxobs'])
            y2 = np.log10(single['fluxmodred'] / single['fluxobs'])
            if single == fuvdata:
                ylabel1 = (r'$\log \Bigg(' + fuvfluxmodintlab + '/' +
                           fuvfluxobslab + '\Bigg)$')
                ylabel2 = (r'$\log \Bigg(' + fuvfluxmodredlab + '/' +
                           fuvfluxobslab + '\Bigg)$')
            elif single == nuvdata:
                ylabel1 = (r'$\log \Bigg(' + nuvfluxmodintlab + '/' +
                           nuvfluxobslab + '\Bigg)$')
                ylabel2 = (r'$\log \Bigg(' + nuvfluxmodredlab + '/' +
                           nuvfluxobslab + '\Bigg)$')
        ylim = [-2.05, 2.05]
        #ylim = [-2, 10]
        plotname = os.path.join(_PLOT_DIR, 'flux_compare_ratio.')
        if not two:
            y1 = y2
            ylabel1 = ylabel2
    else:
        if single is None:
            y1 = np.log10(fuvdata['fluxmodred'])
            y2 = np.log10(nuvdata['fluxmodred'])
            ylabel1 = r'$\log ' + fuvfluxmodredlab
            ylabel2 = r'$\log ' + nuvfluxmodredlab
        else:
            y1 = np.log10(single['fluxmodint'])
            y2 = np.log10(single['fluxmodred'])

        if single == fuvdata:
            ylabel1 = r'$\log ' + fuvfluxmodintlab
            ylabel2 = r'$\log ' + fuvfluxmodredlab
        elif single == nuvdata:
            ylabel1 = r'$\log ' + nuvfluxmodintlab
            ylabel2 = r'$\log ' + nuvfluxmodredlab
        ylim = [-17.4, -11.95]
        plotname = os.path.join(_PLOT_DIR, 'flux_compare.')
        if not two:
            y1 = y2
            ylabel1 = ylabel2

    bg1 = -15.68
    bg2 = -15.48
    xx1 = x1.flatten()
    xx2 = x2.flatten()
    yy1 = y1.flatten()
    yy2 = y2.flatten()
    sel1 = np.isfinite(x1)
    x1 = x1[sel1].flatten()
    y1 = y1[sel1].flatten()
    sel2 = np.isfinite(x2)
    x2 = x2[sel2].flatten()
    y2 = y2[sel2].flatten()

    print np.nanmean(yy1), np.nanmean(yy2)
    print np.nanstd(yy1), np.nanstd(yy2)
    print np.nanmedian(10**yy1), np.nanmedian(10**y2)
    print np.nanstd(10**yy1), np.nanstd(10**yy2)

    #print np.median(y1[x1 > bg1]), np.std(y1[x1 > bg1])
    #print np.median(y2[x2 > bg2]), np.std(y2[x2 > bg2])

    if kwargs['density']:
        func = 'count'
        extend = 'max'
        zlabel = 'Number of Regions'
        zlim = [1, 200]
        cnorm = mcolors.LogNorm(vmin=zlim[0], vmax=zlim[-1])
        fmt = 'log'
        ticknums = [1, 2, 5, 10, 20, 50, 100, 200]
        plotname = plotname.rstrip('.') + '_density.'
    else:
        func = 'median'
        zlabel = r'$A_V + \frac{1}{2}dA_V$'
        extend ='neither'
        zlim = [0, 1.5]
        cnorm = mcolors.Normalize(vmin=zlim[0], vmax=zlim[-1])
        fmt = 'linear'
        ticknums = [np.linspace(zlim[0], zlim[-1], 6)]
        plotname = plotname.rstrip('.') + '_avdav.'

    z = otherdata['avdav'].flatten()#
    hist_kwargs = {'func':func, 'bins':bins, 'xlim':xlim, 'ylim':ylim,
                   'cnorm':cnorm}
    im1, cnorm1, bindata1 = make_2dhist_func(ax1, xx1, yy1, z, cmap=cmapa,
                                             **hist_kwargs)
    if two:
        im2, cnorm2, bindata2 = make_2dhist_func(ax2, xx2, yy2, z, cmap=cmapb,
                                                 **hist_kwargs)
    fmt = None

    if not ratio:
        ax1.plot([-23, 0], [-23, 0], 'k--', lw=2)
        ax2.plot([-23, 0], [-23, 0], 'k--', lw=2)

        selections = region_selections(fuvdata, nuvdata, otherdata, im='flux')
        outline_regions(np.log10(fuvdata['fluxobs']),
                        np.log10(fuvdata['fluxmodred']), selections, ax1)
        outline_regions(np.log10(nuvdata['fluxobs']),
                        np.log10(nuvdata['fluxmodred']), selections, ax2)
    else:
        ax1.plot([-20, -10], [0,0], 'k--', lw=2)
        ax2.plot([-20, -10], [0,0], 'k--', lw=2)

    ## plot running median
    print 'FUV:'
    med_fuv = running_median(x1, y1, ax1, min=-16.85, max=-13.7)
    print 'NUV:'
    med_nuv = running_median(x2, y2, ax2,  min=-16.85, max=-13.7)

    xlabels, ylabels = [xlabel1, xlabel2], [ylabel1, ylabel2]
    xlims, ylims = [xlim, xlim], [ylim, ylim]
    for i, ax in enumerate(fig.axes):
        ax.yaxis.set_ticks(np.arange(np.round(ylims[i][0]), np.round(ylims[i][1])+.5,.5))
        ax.set_xlabel(xlabels[i], fontsize=AXISLABELSIZE)
        ax.set_ylabel(ylabels[i], fontsize=AXISLABELSIZE, labelpad=-5)
        ax.set_xlim(xlims[i])
        ax.set_ylim(ylims[i])
        ax.tick_params('both', labelsize=14)
        ax.yaxis.set_label_coords(-0.18, 0.5)
        #ax.grid(color='w', linewidth=1, alpha=0.5)
        draw_grid(ax=ax)
        #ax.grid(color='k', linewidth=1, alpha=0.5)

    ims = [im1]
    if two:
        ims.append(im2)
    cnorms = [cnorm, cnorm]
    ticks = [ticknums] * 2
    if kwargs['vertical']:
        orientation = 'vertical'
    else:
        orientation = 'horizontal'
    #create_cb(fig, fig.axes, ims, ticks, cnorms, label=['',''],
    #          orientation=orientation, extend=extend, fmt=fmt)

    if two:
        plotname = plotname.rstrip('.') + '_two.'
    print plotname

    if kwargs['save']:
        plt.savefig(plotname + plot_kwargs['format'], **plot_kwargs)
    else:
        plt.show()

def fig5_6(fuvdata, nuvdata, otherdata, xtype='fluxobs', recent=False, color='black', total_bins=9, **kwargs):
    """
    Uncertainties on the modeled flux as a function of observed flux and SFR
    """

    plotname = os.path.join(_PLOT_DIR, 'all_' + xtype + '_uncs.' + plot_kwargs['format'])
    if recent:
        plotname = plotname.replace('_uncs.', '_recent_uncs.')

    if xtype == 'sfr':
        xlabel = r'$\log$ SFR [M$_\odot$ yr$^{-1}$]'
        xdata = otherdata['sfr100']
        sel = [xdata > 0]
        x = np.log10(xdata[sel])
        bg=None
    elif xtype == 'fluxobs':
        xdata1 = fuvdata['fluxobs']
        xdata2 = nuvdata['fluxobs']
        sel1 = [xdata1 > 0]
        sel2 = [xdata2 > 0]
        x1 = np.log10(xdata1[sel1])
        x2 = np.log10(xdata2[sel2])
        xlim = [-16.8, -13.1]
        bg_fuv = -15.53
        bg_nuv = -15.59
    elif xtype == 'avdav':
        xlabel = r'A_V + 1/2 dA_V'
        xdata = otherdata['avdav']
        sel = [xdata > 0]
        x = xdata[sel]

    ylabels = [r'$f^{\mathrm{syn,0}}_{\mathrm{FUV}}$', r'$f^{\mathrm{syn,0}}_{\mathrm{NUV}}$',r'$f^{\mathrm{syn}}_{\mathrm{FUV}}$', r'$f^{\mathrm{syn}}_{\mathrm{NUV}}$']

    fig, ax = plt.subplots(2, 2, figsize=(8,8))

    fluxtypes = ['fuvint', 'nuvint', 'fuvred', 'nuvred']
    for ii, fluxtype in enumerate(fluxtypes):
        if fluxtype[0:3] == 'fuv':
            ind1 = 0
            data = fuvdata
            if xtype == 'fluxobs':
                bg = bg_fuv
                x = x1
                sel = sel1
                xlabel = r'$f^{\mathrm{obs}}_{\mathrm{FUV}}$'
                xlim = [-16.8, -13.2]
                ylim = [-0.55, 1.01]
            elif xtype == 'sfr':
                xlabel = r'SFR [M$_\odot$ yr$^{-1}$]'
                xlim = [-14, -3]
                ylim = [-0.45, 1.19]
                if recent:
                    xlim = [-4.9, -3.2]
                    ylim = [-0.62, 0.5]
                    #xlim = [-6.1, -3.2]
                    #ylim = [-0.5, 0.8]
            color = '#1569C7'
            #color = 'BlueEyes'
        else:
            ind1 = 1
            data = nuvdata
            if xtype == 'fluxobs':
                bg = bg_nuv
                x = x2
                sel = sel2
                xlabel = r'$f^{\mathrm{obs}}_{\mathrm{NUV}}$'
                xlim = [-16.5, -13.03]
                ylim = [-0.55, 1.01]
            elif xtype == 'sfr':
                xlabel = r'SFR [M$_\odot$ yr$^{-1}$]'
                xlim = [-14, -3]
                ylim = [-0.45, 1.19]
                if recent:
                    xlim = [-4.9, -3.2]
                    ylim = [-0.62, 0.5]
                    #xlim = [-6.1, -3.2]
                    #ylim = [-0.5, 0.8]
            color = '#461B7E'
            #color = 'Purple Monster'

        if fluxtype[3:] == 'red':
            best_str = 'fluxmodred'
            ind2 = 1
        else:
            ind2 = 0
            best_str = 'fluxmodint'

        hi_str = best_str + '_upper'
        lo_str = best_str + '_lower'

        y = data[best_str][sel]
        ylo = data[best_str][sel] - data[lo_str][sel]
        yup = data[hi_str][sel] - data[best_str][sel]

        idx, bins, delta = create_bins(x, y, total_bins=total_bins, min=xlim[0], max=xlim[-1])
        ybin_med = np.array([np.nanmean(y[idx==k]) for k in range(1, total_bins+1)])

        for k in range(1, total_bins+1):
            this_upper, this_lower = yup[idx==k], ylo[idx==k]
            this_med = ybin_med[k-1]
            this_lower[this_lower > this_med] = this_med

        bin_upper = np.array([sum_in_quad(yup[idx==k]) for k in range(1, total_bins+1)])
        bin_lower = np.array([sum_in_quad(ylo[idx==k]) for k in range(1, total_bins+1)])

        yerr_upper = np.abs(bin_upper)
        yerr_lower = np.abs(bin_lower)

        plot_x = (bins + delta / 2.)#[:-1]
        plot_y = np.log10(ybin_med)
        plot_ylower = np.log10(ybin_med) - np.log10(ybin_med - yerr_lower)
        plot_yupper = np.log10(ybin_med + yerr_upper) - np.log10(ybin_med)
        plot_ylower[~np.isfinite(plot_ylower)] = 0.0
        #set_trace()
        yerr = [plot_ylower, plot_yupper]
        ylabel = ylabels[ii]
        plot_y = np.zeros(plot_x.shape)

        ax[ind1, ind2].errorbar(plot_x, plot_y, yerr=yerr, marker='o', ms=5,
                                ls='--', color='k', mec='k', lw=0,
                                elinewidth=3, zorder=100, ecolor=color)
        if bg is not None:
            ax[ind1, ind2].fill_between([-20, bg], -2, 2, color='none',
                                        hatch='//', lw=0.7, edgecolor='k')
        draw_grid(ax=ax[ind1, ind2])

        if xtype == 'fluxobs':
            xlabel = xlabel + ' [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]'
        ax[ind1, ind2].set_xlabel(r'$\log$ ' + xlabel, size=16)
        ax[ind1, ind2].set_ylabel(r'$\Delta$ $\log$ ' + ylabel + ' [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]', size=16)

        if xlim is not None:
            ax[ind1, ind2].set_xlim(xlim)
            ax[ind1, ind2].set_ylim(ylim)

    for a in fig.axes:
        a.tick_params(axis='both', labelsize=12)
    for ax in [ax[0,1], ax[1,1]]:
        ax.yaxis.set_label_position("right")
        ax.axes.yaxis.set_ticklabels([])
        ax.yaxis.labelpad = 10

    plt.subplots_adjust(hspace=0.25, wspace=0.05, left=0.1, right=0.9)
    print plotname
    if args.save:
        plt.savefig(plotname, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def fig7(fuvdata, nuvdata, otherdata, log=False, **kwargs):
    """
    Plot maps in observed and intrinsic modeled flux as well as maps of the fraction of obscured star formaion from modeled data and from measured observables.

    """

    ff = 1e-16
    #cnorm1 = mcolors.Normalize(vmin=fluxmin/ff, vmax=fluxmax/ff)
    cnorm1 = mcolors.Normalize(vmin=fluxmin/ff, vmax=fluxmax/ff*5)
    ticks1 = np.linspace(fluxmin/ff, fluxmax/ff, 8)

    ## Fraction obscured using modeled data
    this_sfr = SFR(fuvdata, nuvdata, otherdata)
    sfr_from_mod_int = this_sfr.kennicutt(fuvdata['fluxmodint'],
                                           is_corrected=True)
    sfr_from_obs = this_sfr.kennicutt(fuvdata['fluxobs'], is_corrected=True)

    ratio_mod = sfr_from_obs / sfr_from_mod_int
    ratio_mod = fuvdata['fluxmodred'] / fuvdata['fluxmodint']
    frac_mod = 1. - ratio_mod
    if log:
        frac_mod = np.log10(frac_mod)

    vlims2 = [0.5, np.nanmax(frac_mod)]
    vlims2 = [0.2, 1.0]
    #log=True
    if log:
        vlims2 = [np.log10(x) for x in vlims2]
        cnorm2 = mcolors.Normalize(vmin=vlims2[0], vmax=vlims2[-1])
        ticks2 = np.linspace(vlims2[0], vlims2[-1], 6)
        ticklabels2 = ['{:.2f}'.format((10**x)) for x in ticks2]
    else:
        vlims2 = vlims2
        cnorm2 = mcolors.Normalize(vmin=vlims2[0], vmax=vlims2[1])
        ticks2 = np.linspace(vlims2[0], vlims2[-1], 6)
        #ticklabels = ['{:.3f}'.format((x)) for x in ticks2]
        ticklabels2 = ['{:.2f}'.format((x)) for x in ticks2]


    ## Fraction obscured using 24 micron correction
    sfr_fuv = this_sfr.fuv(fuvdata['fluxobs'])
    sfr_fuv_24 = this_sfr.fuv_24micron(fuvdata['fluxobs'], otherdata['mips24'])
    flux_fuv_24 = sfr_fuv_24[2]

    ratio_obs_a = sfr_from_obs / sfr_fuv_24[0]
    #ratio_obs_b = sfr_from_obs / sfr_fuv_24[1]

    ratio_obs_b = fuvdata['fluxobs'] / flux_fuv_24

    frac_obs_a = 1. - ratio_obs_a
    frac_obs_b = 1. - ratio_obs_b

    get_numbers_obscured_sf(fuvdata, flux_fuv_24)

    reg_paths = get_paths()
    p10 = reg_paths[0].vertices
    p10 = np.vstack([p10, p10[0]])
    p15 = reg_paths[1].vertices
    p15 = np.vstack([p15, p15[0]])
    p05 = reg_paths[2].vertices
    p05 = np.vstack([p05, p05[0]])

    cmap_red = plt.cm.RdYlBu
    cmap_red.set_bad('0.65')

    ## Plot
    rcParams['axes.linewidth'] = 1
    fig, ax = plt.subplots(6, 1, figsize=(5, 9))
    plt.subplots_adjust(left=0.02, right=0.8, bottom=0.02, top=0.98,
                        hspace=0.03)

    im1 = ax[0].imshow((fuvdata['fluxmodred']/ff)[::-1].T,
                       interpolation=interpolation,
                         origin=origin, norm=cnorm1, cmap=cmap)
    ax[0].autoscale(False)
    ax[0].plot(211 - p10[:,1], p10[:,0], color='blue', lw=1)
    ax[0].plot(211 - p15[:,1], p15[:,0], color='purple', lw=1)
    ax[0].plot(211 - p05[:,1], p05[:,0], color='red', lw=1)
    im2 = ax[1].imshow((fuvdata['fluxobs']/ff)[::-1].T, cmap=cmap,
                         origin=origin, interpolation=interpolation,
                         norm=cnorm1)
    im3 = ax[2].imshow((fuvdata['fluxmodint']/ff)[::-1].T, cmap=cmap,
                         origin=origin, interpolation=interpolation,
                         norm=cnorm1)
    im4 = ax[3].imshow((flux_fuv_24[::-1]/ff).T, interpolation=interpolation,
                         origin=origin, norm=cnorm1, cmap=cmap)
    im5 = ax[4].imshow(frac_mod[::-1].T, interpolation=interpolation,
                         origin=origin, norm=cnorm2, cmap=cmap_red)
    im6 = ax[5].imshow(frac_obs_b[::-1].T, interpolation=interpolation,
                         origin=origin, norm=cnorm2, cmap=cmap_red)

    pos1 = ax[0].get_position()
    pos2 = ax[1].get_position()
    pos3 = ax[2].get_position()
    pos4 = ax[3].get_position()
    pos5 = ax[4].get_position()
    pos6 = ax[5].get_position()

    cbax1 = fig.add_axes([pos4.x1 + 0.01, pos4.y0, 0.04, pos1.y1 - pos4.y0])
    cbax2 = fig.add_axes([pos6.x1 + 0.01, pos6.y0, 0.04, pos5.y1 - pos6.y0])
    cbticks = 10

    cb1 = fig.colorbar(im1, cax=cbax1, orientation='vertical', norm=cnorm1, extend='max')
    cb1.ax.tick_params(labelsize=cbticks)
    cb1.set_label(r'Flux $\times 10^{-16}$ [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]', size=12, labelpad=5)

    cb2 = fig.colorbar(im5, cax=cbax2, orientation='vertical', norm=cnorm2, extend='min', ticks=ticks2)
    cb2.ax.tick_params(labelsize=cbticks)
    cb2.set_label(r'Fraction Obscured', size=12, labelpad=5)

    xloc, yloc = 0.01, 0.85
    ax[0].text(xloc, yloc, '(a)', size=12, transform=ax[0].transAxes)
    ax[1].text(xloc, yloc, '(b)', size=12, transform=ax[1].transAxes)
    ax[2].text(xloc, yloc, '(c)', size=12, transform=ax[2].transAxes)
    ax[3].text(xloc, yloc, '(d)', size=12, transform=ax[3].transAxes)
    ax[4].text(xloc, yloc, '(e)', size=12, transform=ax[4].transAxes)
    ax[5].text(xloc, yloc, '(f)', size=12, transform=ax[5].transAxes)

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if kwargs['save']:
        plotname = os.path.join(_PLOT_DIR,'flux_obscured_sf_maps_6vertical_newcmap.'+plot_kwargs['format'])
        plt.savefig(plotname, **plot_kwargs)
    else:
        plt.show()




    fig, ax = plt.subplots(3, 2, figsize=(9,4.2))
    plt.subplots_adjust(left=0.02, right=0.85, bottom=0.02, top=0.98,
                        hspace=0.03, wspace=0.02)
    im1 = ax[0,0].imshow((fuvdata['fluxmodred']/ff)[::-1].T, cmap=cmap,
                          origin=origin, interpolation=interpolation,
                          norm=cnorm1)
    ax[0,0].autoscale(False)
    ax[0,0].plot(211 - p10[:,1], p10[:,0], color='blue', lw=1)
    ax[0,0].plot(211 - p15[:,1], p15[:,0], color='purple', lw=1)
    ax[0,0].plot(211 - p05[:,1], p05[:,0], color='red', lw=1)
    im2 = ax[0,1].imshow((fuvdata['fluxobs']/ff)[::-1].T, cmap=cmap,
                          origin=origin, interpolation=interpolation,
                          norm=cnorm1)
    im3 = ax[1,0].imshow((fuvdata['fluxmodint']/ff)[::-1].T, cmap=cmap,
                          origin=origin, interpolation=interpolation,
                          norm=cnorm1)
    im4 = ax[1,1].imshow((flux_fuv_24/ff)[::-1].T, cmap=cmap, origin=origin,
                         interpolation=interpolation, norm=cnorm1)
    im5 = ax[2,0].imshow(frac_mod[::-1].T, interpolation=interpolation,
                         origin=origin, norm=cnorm2, cmap=cmap_red)
    im6 = ax[2,1].imshow(frac_obs_b[::-1].T, interpolation=interpolation,
                     origin=origin, norm=cnorm2, cmap=cmap_red)

    pos2 = ax[0,1].get_position()
    pos4 = ax[1,1].get_position()
    pos6 = ax[2,1].get_position()

    cbax1 = fig.add_axes([pos4.x1 + 0.01, pos4.y0, 0.02, pos2.y1 - pos4.y0])
    cbax2 = fig.add_axes([pos6.x1 + 0.01, pos6.y0, 0.02, pos6.y1 - pos6.y0])

    cbticks = 10

    cb1 = fig.colorbar(im1, cax=cbax1, orientation='vertical', norm=cnorm1, extend='max')
    cb1.ax.tick_params(labelsize=cbticks)
    cb1.set_label(r'Flux $\times 10^{-16}$ [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]', size=12, labelpad=10)

    cb2 = fig.colorbar(im5, cax=cbax2, orientation='vertical', norm=cnorm2, extend='min')
    cb2.ax.tick_params(labelsize=cbticks)
    cb2.set_label(r'Fraction Obscured', size=12, labelpad=10)

    xloc, yloc = 0.01, 0.85
    ax[0,0].text(xloc, yloc, '(a)', size=12, transform=ax[0,0].transAxes)
    ax[0,1].text(xloc, yloc, '(b)', size=12, transform=ax[0,1].transAxes)
    ax[1,0].text(xloc, yloc, '(c)', size=12, transform=ax[1,0].transAxes)
    ax[1,1].text(xloc, yloc, '(d)', size=12, transform=ax[1,1].transAxes)
    ax[2,0].text(xloc, yloc, '(e)', size=12, transform=ax[2,0].transAxes)
    ax[2,1].text(xloc, yloc, '(f)', size=12, transform=ax[2,1].transAxes)

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if kwargs['save']:
        plotname = os.path.join(_PLOT_DIR,'flux_obscured_sf_maps_6tile_newcmap.'+plot_kwargs['format'])
        plt.savefig(plotname, **plot_kwargs)
    else:
        plt.show()

    # for ax in fig.axes:
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # if kwargs['save']:
    #     plotname = os.path.join(_PLOT_DIR,'flux_obscured_sf_maps.'+plot_kwargs['format'])
    #     plt.savefig(plotname, **plot_kwargs)
    # else:
    #     plt.show()
    rcParams['axes.linewidth'] = 2

def fig8(fuvdata, nuvdata, otherdata, **kwargs):
    """
    Take the observed flux and do the usual dust corrections then convert to SFR with Kennicutt conversions and compare with the CMD-based SFR.
    This may help us to see whether the widely used dust corrections are correct. Is there a trend?

    astrogrid.flux.galex_flux2mag(fuvfluxobs, 'galex_fuv')

    """
    from scipy.optimize import curve_fit
    from numpy.polynomial import polynomial as P

    cmd_sfr = otherdata['sfr100']

    ## Make SFR object
    this_sfr = SFR(fuvdata, nuvdata, otherdata)

    sfr_from_obs_col = this_sfr.kennicutt(fuvdata['fluxobs'], is_corrected=False, ben=False, hao=False, color=True, modeled=False)

    sfr_from_fuv_24micron = this_sfr.fuv_24micron(fuvdata['fluxobs'], otherdata['mips24'])

    sfr_from_mod = this_sfr.kennicutt(fuvdata['fluxmodint'], is_corrected=True, ben=False, hao=False, color=False, modeled=True)


    x = np.log10(cmd_sfr).flatten()
    y1 = np.log10(sfr_from_fuv_24micron)[1].flatten()
    y2 = np.log10(sfr_from_obs_col).flatten()
    y3 = np.log10(sfr_from_fuv_24micron)[0].flatten()
    x1 = np.log10(sfr_from_mod).flatten()

    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A * x + B

    A, B = curve_fit(f, x[x > -5], y1[x > -5])[0]
    C, D = curve_fit(f, x[x > -5], y1[x > -5], absolute_sigma=False, sigma=fuvdata['fluxobs'].flatten()[x > -5])[0]
    xvals = np.linspace(-12, 0, 1000)
    yvals = np.linspace(0, 0, 1000)
    yvals1 = xvals + (B + 1.57)
    yvals2 = A * xvals + B

    c, stats = P.polyfit(x[x > -5],y1[x > -5],1,full=True, w=sfr_from_fuv_24micron[2].flatten()[x > -5])
    yvals3 = c[0] + xvals * c[1]
    yvals4 = C * xvals + D
    #set_trace()

    y1a = np.log10(sfr_from_fuv_24micron[1] / cmd_sfr).flatten()
    y2a = np.log10(sfr_from_obs_col / cmd_sfr).flatten()
    y3a = np.log10(sfr_from_fuv_24micron[0] / cmd_sfr).flatten()

    #y1a = np.log10(sfr_from_fuv_24micron[1] - cmd_sfr).flatten()
    #y2a = np.log10(sfr_from_obs_col - cmd_sfr).flatten()
    #y3a = np.log10(sfr_from_fuv_24micron[0] - cmd_sfr).flatten()
    #set_trace()
    sel = (np.isfinite(y1) & np.isfinite(x))
    xx = x[sel]
    yy1, yy2, yy3, xx1 = y1[sel], y2[sel], y3[sel], x1[sel]

    sela = (np.isfinite(y1a) & np.isfinite(x))
    xx = x[sel]
    yy1a, yy2a, yy3a = y1a[sela], y2a[sela], y3a[sela]

    xmean = np.mean(x[np.isfinite(x)])
    #ymeans1 = [np.nanmean(thisy[np.isfinite(thisy)]) for thisy in [y1, y2, y3]]

    #flux-weighted means
    w = sfr_from_fuv_24micron[2].flatten()
    ww = fuvdata['fluxobs'].flatten()
    xmeans = np.average(x[np.isfinite(x)], weights=ww[np.isfinite(x)])
    xmeans1 = np.average(x1[np.isfinite(x1)], weights=ww[np.isfinite(x1)])
    ymeans2 = [np.average(thisy[np.isfinite(thisy)], weights=ww[np.isfinite(thisy)]) for thisy in [y1, y2, y3]]

    sx = np.where((xvals > xmeans-0.005) & (xvals < xmeans+0.005))
    sy = np.where((xvals > ymeans2[0]-0.005) & (xvals < ymeans2[0]+0.005))
    print xmeans, ymeans2[0]
    print xvals[sx], xvals[sy]
    print xmeans - xvals[sy]
    print ymeans2[0] - xvals[sx]
    #exit()
    distances = [np.sqrt((foo-xmeans)**2 + (foo-ymeans2[0]**2)) for foo in xvals]
    #set_trace()
    xmed = np.median(xx[np.isfinite(xx)])
    ymeds = [np.nanmedian(thisy[np.isfinite(thisy)]) for thisy in [yy1, yy2, yy3]]

    cmap = plt.cm.Greens_r
    cmap.set_bad('white')


    xlim = [-10, -3.05]#[-14.1, -2.5]
    ylim = [-6.9, -2.9]# 3.5]
    ylima = [-4, 8]
    #ylima = [-9, 3]
    #ylima = [-0.01, 0.01]
    vmin, vmax = 1, 200
    bins = 50

    ts = 12
    xs, ys = 0.97, 0.05


    #set_trace()
    #fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(8,8))
    ax = [ax1, ax2, ax3, ax4]

    im1, cnorm1, h1, xe1, ye1 = make_2dhist(ax[0], x, y1, bins, xlim=xlim,
                                            ylim=ylim, vmin=vmin, vmax=vmax,
                                            cmap=cmap)
    ax[0].plot(xmeans, ymeans2[0], 'k*', ms=15, zorder=101)
    ax[0].text(xs, ys, r'$24\,\mu$m corrected', transform=ax[0].transAxes,
               size=14, ha='right')
    #med_fuv_24micron = running_median(x, y1, ax[0], min=-10, max=-3, color='blue', flux_ratio=False)
    ax[0].plot(xvals, xvals, 'k--', lw=2)
    #ax[0].plot(xvals[sx], xvals[sy], 'yo', ms=5, mec='y')
    #ax[0].plot([ymeans2[0], xvals[sx]], [ymeans2[0], ymeans2[0]], color='r')
    #ax[0].plot([xmeans, xmeans], [ymeans2[0], xmeans], color='b')
    #ax[0].plot(xvals, yvals1, color='DarkViolet', ls='--', lw=3)
    #ax[0].plot(xvals, yvals2, color='DarkViolet', ls=':', lw=3)
    #ax[0].plot(xvals, xvals + B + 1.67, color='blue', ls='--', lw=1)
    #ax[0].plot(xvals, yvals4, color='orange', ls='--', lw=3)


    im2, cnorm2, h2, xe2, ye2 = make_2dhist(ax[1], x, y2, bins, xlim=xlim,
                                            ylim=ylim, vmin=vmin, vmax=vmax,
                                            cmap=cmap)
    ax[1].plot(xmeans, ymeans2[1], 'k*', ms=15, zorder=101)
    ax[1].text(xs, ys, 'FUV - NUV\ncorrected', transform=ax[1].transAxes,
               size=14, ha='right', ma='center', va='bottom')
    #med_color = running_median(x, y2, ax[1], min=-10, max=-3, color='blue', flux_ratio=False)


    ff = 1
    im3, cnorm3, h3, xe3, ye3 = make_2dhist(ax[2], x, y1a*ff, bins, xlim=xlim,
                                            ylim=ylima, vmin=vmin, vmax=vmax,
                                            cmap=cmap)
    #ax[2].plot(xmeans, ymeans2[1], 'k*', ms=15)
    ax[2].text(xs, ys, r'$24\,\mu$m corrected', transform=ax[2].transAxes,
               size=14, ha='right')
    med_color = running_median(x, y1a*ff, ax[2], min=-10, max=-3, color='blue', flux_ratio=False)
    ax[2].plot([-12, -1], [0,0], 'k--', lw=3)
    ax[2].plot(xmeans, np.log10(10**ymeans2[0]/10**xmeans), 'k*', ms=15, zorder=101)

    im4, cnorm4, h4, xe4, ye4 = make_2dhist(ax[3], x, y2a*ff, bins, xlim=xlim,
                                            ylim=ylima, vmin=vmin, vmax=vmax,
                                            cmap=cmap)
    #ax[3].plot(xmeans, ymeans2[1], 'k*', ms=15, zorder=10)
    ax[3].text(xs, ys, 'FUV - NUV\ncorrected', transform=ax[3].transAxes,
               size=14, ha='right', ma='center', va='bottom')
    med_color = running_median(x, y2a*ff, ax[3], min=-10, max=-3, color='blue', flux_ratio=False)
    ax[3].plot([-12, -1], [0,0], 'k--', lw=3)
    ax[3].plot(xmeans, np.log10(10**ymeans2[1]/10**xmeans), 'k*', ms=15, zorder=101)



    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[1].set_ylim(ylim)
    ax[1].set_yticklabels([])
    ax[2].set_ylim(ylima)
    ax[3].set_ylim(ylima)
    ax[3].set_yticklabels([])
    ax[0].set_ylabel(r'$\log$ SFR$_{\textrm{\small Flux}}$ [M$_\odot$ yr$^{-1}$]', size=18)
    ax[2].set_ylabel(r'$\log$(SFR$_{\textrm{\small Flux}}$ / SFR$_{\textrm{\small CMD}}$)', size=18)

    for ax in fig.axes:
        draw_grid(ax=ax)
        ax.tick_params(axis='both', labelsize=14)
    for ax in [ax1, ax2]:
        ax.plot(xvals, xvals, 'k--', lw=3)

    fig.text(0.55, 0.02, r'$\log$ SFR$_{\textrm{\small CMD}}$ [M$_\odot$ yr$^{-1}$]',
             size=18, ha='center')
    #fig.text(0.02, 0.55, r'$\log$ Flux-based SFR [M$_\odot$ yr$^{-1}$]',
    #         size=18, va='center', rotation='vertical')

    #plt.subplots_adjust(wspace=0.04, bottom=0.15, left=0.08, top=0.97, right=0.97)

    plt.subplots_adjust(right=0.98, top=0.98, wspace=0.02, hspace=0.02, bottom=0.1, left=0.1)

    if args.save:
        plotname = os.path.join(_PLOT_DIR, 'sfr_correct_compare_4.')
        plt.savefig(plotname + plot_kwargs['format'], **plot_kwargs)
    else:
        plt.show()


def fig9_10_11(fuvdata, nuvdata, ratio=True, two=True, single=None, **kwargs):
    """
    f^syn_red/f^obs vs f_obs.
    Ratio of modeled reddened flux to observed flux as a function of observed flux. Also plots running median.
    """
    check=False
    plt.close('all')
    fig1, ax1 = plt.subplots(2, 4, figsize=(8,4), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.04, wspace=0.04, top=0.95, bottom=0.16,
                        left=0.14, right=0.95)

    fig2, ax2 = plt.subplots(2, 4, figsize=(8,4), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.04, wspace=0.04, top=0.95, bottom=0.16,
                        left=0.14, right=0.95)

    xlim1 = [-17.8, -12.2]
    ylim1 = [-2.5, 2.5]
    xlim2 = [-17.4, -12.2]
    ylim2 = [-2.5, 1.45]
    fluxunits = r'  [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]'
    xlabel1 = r'$\log$ ' + fuvfluxobslab + fluxunits
    xlabel2 = r'$\log$ ' + nuvfluxobslab + fluxunits
    ylabel1 = (r'$\log \Bigg(' + fuvfluxmodredlab + '/' +
               fuvfluxobslab + '\Bigg)$')
    ylabel2 = (r'$\log \Bigg(' + nuvfluxmodredlab + '/' +
               nuvfluxobslab + '\Bigg)$')

    cmapa, cmapb = cmap2, cmap3
    cmapa.set_bad('white')
    cmapb.set_bad('white')

    func = 'count'
    extend = 'max'
    zlabel = 'Number of Regions'
    zlim = [1, 200]
    cnorm = mcolors.LogNorm(vmin=zlim[0], vmax=zlim[-1])
    fmt = 'log'
    ticknums = [1, 2, 5, 10, 20, 50, 100, 200]
    z = otherdata['sfr100'].flatten()#
    hist_kwargs1 = {'func':func, 'bins':bins, 'xlim':xlim1, 'ylim':ylim1,
                   'cnorm':cnorm}
    hist_kwargs2 = {'func':func, 'bins':bins, 'xlim':xlim2, 'ylim':ylim2,
                   'cnorm':cnorm}

    mdata = ['fluxmodred', 'fluxmodred5000', 'fluxmodred1000',
             'fluxmodred750', 'fluxmodred500', 'fluxmodred300',
             'fluxmodred200', 'fluxmodred100']

    lims = [r'\textbf{Full}', r'\textbf{5 Gyr}', r'\textbf{1 Gyr}', r'\textbf{750 Myr}', r'\textbf{500 Myr}', r'\textbf{300 Myr}', r'\textbf{200 Myr}', r'\textbf{100 Myr}']
    for ii, md in enumerate(mdata):
        ix = ii/4
        iy = ii%4

        y1 = np.log10(fuvdata[md] / fuvdata['fluxobs'])
        y2 = np.log10(nuvdata[md] / nuvdata['fluxobs'])

        x1 = np.log10(fuvdata['fluxobs'])
        x2 = np.log10(nuvdata['fluxobs'])
        sel1 = np.isfinite(y1)
        sel2 = np.isfinite(y2)
        xx1 = x1.flatten()
        xx2 = x2.flatten()
        x1 = x1[sel1].flatten()
        x2 = x2[sel2].flatten()
        yy1 = y1.flatten()
        yy2 = y2.flatten()
        y1 = y1[sel1].flatten()
        y2 = y2[sel2].flatten()


        flux_percent_fuv = np.around(np.nansum(fuvdata[md]) / np.nansum(fuvdata['fluxobs']), 2)
        flux_percent_nuv = np.around(np.nansum(nuvdata[md]) / np.nansum(nuvdata['fluxobs']), 2)

        #flux_percent_fuv = np.around(np.nanmean(y1[np.isfinite(y1)].flatten()), 2)
        #flux_percent_nuv = np.around(np.nanmean(y2[np.isfinite(y2)].flatten()), 2)
        #print np.nanmean(yy1), np.nanmean(yy2)
        #print np.nanstd(yy1), np.nanstd(yy2)

        agelim = lims[ii]

        im1, cnorm1, bindata1 = make_2dhist_func(ax1[ix,iy], xx1, yy1, z,
                                                 cmap=cmapa, **hist_kwargs1)
        im2, cnorm2, bindata2 = make_2dhist_func(ax2[ix,iy], xx2, yy2, z,
                                                 cmap=cmapb, **hist_kwargs2)
        fmt = None

        ax1[ix,iy].plot([-20, -10], [0,0], 'k--', lw=2)
        ax2[ix,iy].plot([-20, -10], [0,0], 'k--', lw=2)

        ax1[ix,iy].text(0.95, 0.95, agelim, fontsize=12, ha='right', va='top', transform=ax1[ix,iy].transAxes)
        ax2[ix,iy].text(0.95, 0.95, agelim, fontsize=12, ha='right', va='top', transform=ax2[ix,iy].transAxes)
        ax1[ix,iy].text(0.95, 0.82, flux_percent_fuv, fontsize=12, ha='right', va='top', transform=ax1[ix,iy].transAxes)
        ax2[ix,iy].text(0.95, 0.82, flux_percent_nuv, fontsize=12, ha='right', va='top', transform=ax2[ix,iy].transAxes)


        ## plot running median
        print 'FUV:'
        med_fuv = running_median(x1, y1, ax1[ix,iy], min=-16.85, max=-13.7, check=check)
        print 'NUV:'
        med_nuv = running_median(x2, y2, ax2[ix,iy],  min=-16.85, max=-13.7, check=check)

    xlabels, ylabels = [xlabel1, xlabel2], [ylabel1, ylabel2]
    xlims, ylims = [xlim1, xlim2], [ylim1, ylim2]
    fig_nums = plt.get_fignums()
    for jj, fi in enumerate(fig_nums):
        this_fig = plt.figure(fi)
        this_fig.text(0.5, 0.02, xlabels[jj], fontsize=AXISLABELSIZE, ha='center')
        this_fig.text(0.02, 0.5, ylabels[jj], fontsize=AXISLABELSIZE, va='center', rotation='vertical')
        xlim = xlims[jj]
        ylim = ylims[jj]
        #set_trace()
        #plt.xlabel(xlabels[jj], fontsize=AXISLABELSIZE)
        #plt.ylabel(ylabels[jj], fontsize=AXISLABELSIZE)
        for ax in this_fig.axes:
            #set_trace()
            ax.yaxis.set_ticks(np.arange(np.round(ylim[0]), np.round(ylim[1])+.5,.5))
            #set_trace()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params('both', labelsize=11)
            #ax.yaxis.set_label_coords(-0.18, 0.5)
            draw_grid(ax=ax)

        #ims = [im1, im2]
        #cnorms = [cnorm, cnorm]
        #ticks = [ticknums] * 2
        #orientation = 'vertical'
        #create_cb(fig, fig.axes, ims, ticks, cnorms, label=['',''],
        #          orientation=orientation, extend=extend, fmt=fmt)

        plotname = os.path.join(_PLOT_DIR, 'flux_compare_ratio_agelimits.')
        if fi == 1: plotname = plotname.replace('.', '_fuv.')
        if fi == 2: plotname = plotname.replace('.', '_nuv.')
        if kwargs['save']:
            plt.savefig(plotname + plot_kwargs['format'], **plot_kwargs)
        else:
            plt.show()

def fig12(fuvdata, nuvdata, otherdata, **kwargs):
    """
    Take the observed flux and do the usual dust corrections then convert to SFR with Kennicutt conversions and compare with the CMD-based SFR.
    This may help us to see whether the widely used dust corrections are correct. Is there a trend?

    astrogrid.flux.galex_flux2mag(fuvfluxobs, 'galex_fuv')

    """
    from scipy.optimize import curve_fit
    from numpy.polynomial import polynomial as P

    cmd_sfr = otherdata['sfr100']

    ## Make SFR object
    this_sfr = SFR(fuvdata, nuvdata, otherdata)

    sfr_from_obs_col = this_sfr.kennicutt(fuvdata['fluxobs'], is_corrected=False, ben=False, hao=False, color=True, modeled=False)

    sfr_from_fuv_24micron = this_sfr.fuv_24micron(fuvdata['fluxobs'], otherdata['mips24'])

    sfr_from_mod = this_sfr.kennicutt(fuvdata['fluxmodint'], is_corrected=True, ben=False, hao=False, color=False, modeled=True)


    x = np.log10(cmd_sfr).flatten()
    y1 = np.log10(sfr_from_fuv_24micron)[1].flatten()
    y2 = np.log10(sfr_from_obs_col).flatten()
    y3 = np.log10(sfr_from_fuv_24micron)[0].flatten()
    x1 = np.log10(sfr_from_mod).flatten()

    def f(x, A, B): # this is your 'straight line' y=f(x)
        return A * x + B

    xvals = np.linspace(-12, 0, 1000)
    yvals = np.linspace(0, 0, 1000)

    y1a = np.log10(sfr_from_fuv_24micron[1] / cmd_sfr).flatten()
    y2a = np.log10(sfr_from_obs_col / cmd_sfr).flatten()
    y3a = np.log10(sfr_from_fuv_24micron[0] / cmd_sfr).flatten()


    sel = (np.isfinite(y1) & np.isfinite(x))
    xx = x[sel]
    yy1, yy2, yy3, xx1 = y1[sel], y2[sel], y3[sel], x1[sel]

    sela = (np.isfinite(y1a) & np.isfinite(x))
    xx = x[sel]
    yy1a, yy2a, yy3a = y1a[sela], y2a[sela], y3a[sela]

    xmean = np.mean(x[np.isfinite(x)])

    #flux-weighted means
    w = sfr_from_fuv_24micron[2].flatten()
    ww = fuvdata['fluxobs'].flatten()
    xmeans = np.average(x[np.isfinite(x)], weights=ww[np.isfinite(x)])
    xmeans1 = np.average(x1[np.isfinite(x1)], weights=ww[np.isfinite(x1)])
    ymeans2 = [np.average(thisy[np.isfinite(thisy)], weights=ww[np.isfinite(thisy)]) for thisy in [y1, y2, y3]]

    sx = np.where((xvals > xmeans-0.01) & (xvals < xmeans+0.01))
    sy = np.where((xvals > ymeans2[0]-0.01) & (xvals < ymeans2[0]+0.01))
    print xmeans, ymeans2[0]
    print np.mean(xvals[sx]), np.mean(xvals[sy])
    print xmeans - np.mean(xvals[sy])
    print ymeans2[0] - np.mean(xvals[sx])

    distances = [np.sqrt((foo-xmeans)**2 + (foo-ymeans2[0]**2)) for foo in xvals]

    xmed = np.median(xx[np.isfinite(xx)])
    ymeds = [np.nanmedian(thisy[np.isfinite(thisy)]) for thisy in [yy1, yy2, yy3]]

    cmap = plt.cm.Greens_r
    cmap.set_bad('white')


    xlim = [-10, -3.05]#[-14.1, -2.5]
    ylim = [-6.9, -2.9]# 3.5]
    ylima = [-4, 8]
    vmin, vmax = 1, 200
    bins = 50

    ts = 12
    xs, ys = 0.97, 0.05

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, figsize=(8,8))
    ax = [ax1, ax2, ax3, ax4]

    im1, cnorm1, h1, xe1, ye1 = make_2dhist(ax[0], x, y1, bins, xlim=xlim,
                                            ylim=ylim, vmin=vmin, vmax=vmax,
                                            cmap=cmap)
    ax[0].plot(xmeans, ymeans2[0], 'k*', ms=15, zorder=101)
    ax[0].text(xs, ys, r'$24\,\mu$m corrected', transform=ax[0].transAxes,
               size=14, ha='right')
    ax[0].plot(xvals, xvals, 'k--', lw=2)

    im2, cnorm2, h2, xe2, ye2 = make_2dhist(ax[1], x, y2, bins, xlim=xlim,
                                            ylim=ylim, vmin=vmin, vmax=vmax,
                                            cmap=cmap)
    ax[1].plot(xmeans, ymeans2[1], 'k*', ms=15, zorder=101)
    ax[1].text(xs, ys, 'FUV - NUV\ncorrected', transform=ax[1].transAxes,
               size=14, ha='right', ma='center', va='bottom')

    im3, cnorm3, h3, xe3, ye3 = make_2dhist(ax[2], x, y1a, bins, xlim=xlim,
                                            ylim=ylima, vmin=vmin, vmax=vmax,
                                            cmap=cmap)
    ax[2].text(xs, ys, r'$24\,\mu$m corrected', transform=ax[2].transAxes,
               size=14, ha='right')
    med_color = running_median(x, y1a, ax[2], min=-10, max=-3, color='blue', flux_ratio=False)
    ax[2].plot([-12, -1], [0,0], 'k--', lw=3)
    ax[2].plot(xmeans, np.log10(10**ymeans2[0]/10**xmeans), 'k*', ms=15, zorder=101)

    im4, cnorm4, h4, xe4, ye4 = make_2dhist(ax[3], x, y2a, bins, xlim=xlim,
                                            ylim=ylima, vmin=vmin, vmax=vmax,
                                            cmap=cmap)
    ax[3].text(xs, ys, 'FUV - NUV\ncorrected', transform=ax[3].transAxes,
               size=14, ha='right', ma='center', va='bottom')
    med_color = running_median(x, y2a, ax[3], min=-10, max=-3, color='blue', flux_ratio=False)
    ax[3].plot([-12, -1], [0,0], 'k--', lw=3)
    ax[3].plot(xmeans, np.log10(10**ymeans2[1]/10**xmeans), 'k*', ms=15, zorder=101)



    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[1].set_ylim(ylim)
    ax[1].set_yticklabels([])
    ax[2].set_ylim(ylima)
    ax[3].set_ylim(ylima)
    ax[3].set_yticklabels([])
    ax[0].set_ylabel(r'$\log$ SFR$_{\textrm{\small Flux}}$ [M$_\odot$ yr$^{-1}$]', size=18)
    ax[2].set_ylabel(r'$\log$(SFR$_{\textrm{\small Flux}}$ / SFR$_{\textrm{\small CMD}}$)', size=18)

    for ax in fig.axes:
        draw_grid(ax=ax)
        ax.tick_params(axis='both', labelsize=14)
    for ax in [ax1, ax2]:
        ax.plot(xvals, xvals, 'k--', lw=3)

    fig.text(0.55, 0.02, r'$\log$ SFR$_{\textrm{\small CMD}}$ [M$_\odot$ yr$^{-1}$]',
             size=18, ha='center')
    #fig.text(0.02, 0.55, r'$\log$ Flux-based SFR [M$_\odot$ yr$^{-1}$]',
    #         size=18, va='center', rotation='vertical')

    #plt.subplots_adjust(wspace=0.04, bottom=0.15, left=0.08, top=0.97, right=0.97)

    plt.subplots_adjust(right=0.98, top=0.98, wspace=0.02, hspace=0.02, bottom=0.1, left=0.1)

    if args.save:
        plotname = os.path.join(_PLOT_DIR, 'sfr_correct_compare_old_correct.')
        plt.savefig(plotname + plot_kwargs['format'], **plot_kwargs)
    else:
        plt.show()


def uncertainty(data, flux_str):
    ftype = 'reddened' if flux_str[-3:] == 'red' else 'dust-free'
    datatype = 'FUV' if np.array_equal(data, fuvdata) else 'NUV'

    x = data[flux_str]
    x_lower = data[flux_str + '_lower']
    x_upper = data[flux_str + '_upper']
    y = data['fluxobs']

    z = np.log10(x / y)
    z = z[np.isfinite(z)]
    z_upper = np.log10(x_upper / y)
    z_lower = np.log10(x_lower / y)
    z_upper = z_upper[np.isfinite(z_upper)]
    z_lower = z_lower[np.isfinite(x_lower)]
    lower = np.sqrt(np.sum((z_lower - z)**2) / len(z))
    upper = np.sqrt(np.sum((z_upper - z)**2) / len(z))

    print 'Uncertainty analysis'
    print ' Ratio of modeled, ' + ftype + ' to observed ' + datatype + ' flux:'
    print '     Mean:   ', np.mean(z)
    print '     Median: ', np.median(z)
    print '     std:    ', np.std(z)
    print '     Upper SFH error: ', upper
    print '     Lower SFH error: ', lower




if __name__ == '__main__':
    sns.reset_orig()
    args = get_args()
    res = args.res
    dust_curve = args.dust_curve

    kwargs = {'density': args.density, 'outline': args.outline, 'save': args.save, 'red': args.red, 'sfhs': args.sfhs, 'vertical': True, 'ybar_outline': args.ybar_outline}
    plot_kwargs = {'dpi':300, 'bbox_inches':'tight', 'format':'pdf'}

    if os.environ['PATH'][1:6] == 'astro':
        _TOP_DIR = '/astro/store/phat/arlewis/'
    else:
        _TOP_DIR = '/Users/alexialewis/research/PHAT/'

    _DATA_DIR = os.path.join(_TOP_DIR, 'maps', 'analysis')
    _WORK_DIR = os.path.join(_TOP_DIR, 'uvflux')
    _PLOT_DIR = os.path.join(_WORK_DIR, 'plots')

    CBARTICKSIZE, AXISLABELSIZE, PLOTTEXTSIZE = 12, 18, 16

    sfr100lab = r'\textbf{$\langle\mathrm{SFR}\rangle_{100}$}'
    fuvfluxobslab = r'\textbf{$f_\mathrm{FUV}^\mathrm{obs}$}'
    fuvfluxmodredlab = r'\textbf{$f_\mathrm{FUV}^\mathrm{syn}$}'
    fuvfluxmodintlab = r'\textbf{$f_\mathrm{FUV}^\mathrm{syn,0}$}'

    nuvfluxobslab = r'\textbf{$f_\mathrm{NUV}^\mathrm{obs}$}'
    nuvfluxmodredlab = r'\textbf{$f_\mathrm{NUV}^\mathrm{syn}$}'
    nuvfluxmodintlab = r'\textbf{$f_\mathrm{NUV}^\mathrm{syn,0}$}'
    matchdustlab = r'$A_V + dA_V/2$'

    cmap, cmap2, cmap3, cmap4, cmap_resid = set_colormaps() #heat, blue, purple, green
    origin = 'lower'
    aspect = 'auto'
    bins = 75
    interpolation='nearest'#none'

    fluxmin, fluxmin2, fluxmax, fluxmaxint = 1e-20, 1e-16, 4e-15, 4e-14
    matchdustlim = [-0.06, 1.52]
    sfrmin, sfrmax = 1e-6, 3e-4

    fuvdata, nuvdata, otherdata = compile_data.gather_map_data(res, dust_curve,
                                                               sfh=args.sfh)
    sfhcube, sfhcube_upper, sfhcube_lower,sfhhdr = compile_data.gather_sfh(res)


    #fig1(**kwargs)
    #fig2_3(fuvdata, **kwargs)
    #fig2_3(nuvdata, **kwargs)
    #fig4(fuvdata, nuvdata, ratio=True, two=True, single=None, **kwargs)
    #fig5_6(fuvdata, nuvdata, otherdata, xtype='fluxobs')
    #fig5_6(fuvdata, nuvdata, otherdata, xtype='sfr', recent=True)
    fig7(fuvdata, nuvdata, otherdata, **kwargs)
    #fig8(fuvdata, nuvdata, otherdata, **kwargs)


    #fuvdata, nuvdata, otherdata = compile_data.gather_map_data_agelim(res, dust_curve, sfh=args.sfh, correct_obs=False)
    #fig9_10_11(fuvdata, nuvdata, **kwargs)

    #fuvdata, nuvdata, otherdata = compile_data.gather_map_data_agelim(res, dust_curve, sfh=args.sfh, correct_obs=True)
    #fig9_10_11(fuvdata, nuvdata, **kwargs)
    #fig12(fuvdata, nuvdata, otherdata, **kwargs)

    #[uncertainty(data, fluxtype) for data in [fuvdata, nuvdata] for fluxtype in ['fluxmodred', 'fluxmodint']]



