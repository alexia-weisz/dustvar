import numpy as np
import astropy.io.fits as pyfits
import astropy.units as u
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import astrogrid
from matplotlib.ticker import ScalarFormatter, LogFormatter
from pdb import set_trace
import plot_extcurves



def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')

    return parser.parse_args()

def create_cb(fig, axes_list, ims, ticks, cnorms, fmt='log', label='',
              labsize=14, pad=None):
    for i, ax in enumerate(axes_list):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y1 + 0.02, pos.x1-pos.x0, 0.03])
        if fmt == 'log':
            cform = LogFormatter(base=10, labelOnlyBase=False)
        else:
            cform = ScalarFormatter(useOffset=False)
        cb = fig.colorbar(ims[i], cax=cax, orientation='horizontal',
                          norm=cnorms[i], format=cform, ticks=ticks[i])
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=CBARTICKSIZE)
        cb.set_label(label[i], size=labsize, labelpad=pad)


def make_2dhist(ax, x, y, bins, xlim=None, ylim=None, vmin=1, vmax=500,
                origin='lower', aspect='auto', cmap=plt.cm.Blues):
    if xlim is None and ylim is None:
        kwargs = {'bins': bins}
    else:
        kwargs = {'bins': bins, 'range': [xlim, ylim]}
    h, xe, ye = np.histogram2d(x, y, **kwargs)
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    cnorm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(h.T, cmap=cmap, interpolation='none', extent=extent,
                   aspect=aspect, norm=cnorm)
    return im, cnorm


def plot_3im(fuvdata, nuvdata, bins=75, figsize=(12,5)):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    obs = fuvdata['magobs'] - nuvdata['magobs']
    mod = fuvdata['magmodint'] - nuvdata['magmodint']
    x = avdav
    y1 = (obs) - (mod)
    y2 = np.log10(fuvdata['fluxmodint']/fuvdata['fluxobs'])
    y3 = np.log10(nuvdata['fluxmodint']/nuvdata['fluxobs'])

    sel = np.isfinite(x)
    x = x[sel].flatten()
    y1 = y1[sel].flatten()
    y2 = y2[sel].flatten()
    y3 = y3[sel].flatten()

    xlim = matchdustlim
    ylim1 = [-2.2, 6.1]
    ylim2 = [-1.1, 2.7]
    ylim3 = [-1.35, 2.7]

    xlabel = matchdustlab
    ylabel1 = r'$(\mathrm{FUV} - \mathrm{NUV})_\mathrm{obs} - (\mathrm{FUV} - \mathrm{NUV})_\mathrm{SFH,0}$'
    ylabel2 = (r'$\log \Bigg(' + fuvfluxmodintlab + '/' + fuvfluxobslab +
               '\Bigg)$')
    ylabel3 = (r'$\log \Bigg(' + nuvfluxmodintlab + '/' + nuvfluxobslab +
               '\Bigg)$')

    im1, cnorm1 = make_2dhist(ax1, x, y1, bins, xlim=xlim, ylim=ylim1,
                              vmin=1, vmax=60, cmap=cmap4)
    im2, cnorm2 = make_2dhist(ax2, x, y2, bins, xlim=xlim, ylim=ylim2,
                              vmin=1, vmax=200, cmap=cmap2)
    im3, cnorm3 = make_2dhist(ax3, x, y3, bins, xlim=xlim, ylim=ylim3,
                              vmin=1, vmax=200, cmap=cmap3)


    plt.subplots_adjust(wspace=1./figsize[0]*5.5, top=.8,
                        bottom=0.15, right=0.77, left=0.07)

    cnorms = [cnorm1, cnorm2, cnorm3]
    ims = [im1, im2, im3]
    ticks = [[1, 2, 5, 10, 20, 50, 100]] * len(cnorms)
    cblabels = [' '] * len(cnorms)
    xlims = [xlim] * 3
    ylims = [ylim1, ylim2, ylim3]
    xlabels = [xlabel] * 3
    ylabels = [ylabel1, ylabel2, ylabel3]
    
    create_cb(fig, [ax1, ax2, ax3], ims, ticks, cnorms, label=cblabels)
    
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.tick_params(axis='both', labelsize=13)
        ax.set_xlim(xlims[i])
        ax.set_ylim(ylims[i])
        ax.set_xlabel(xlabels[i], fontsize=AXISLABELSIZE)
        ax.set_ylabel(ylabels[i], fontsize=15)

    return ax1, ax2, ax3


def make_plots(fuvdata, nuvdata, plot_laws, plotfile=None, save=False):
    ax1, ax2, ax3 = plot_3im(fuvdata, nuvdata)
    plot_extcurves.plot_curve(ax1, plot_laws=plot_laws, plot_type='color')
    plot_extcurves.plot_curve(ax2, plot_laws=plot_laws, plot_type='flux')
    plot_extcurves.plot_curve(ax3, plot_laws=plot_laws, plot_type='flux',
                              legend=True)
    
    if save:
        plt.savefig(plotfile, **plot_kwargs)
        plt.close()
    else:
        plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    
args = get_args()

plot_kwargs = {'dpi':300, 'bbox_inches':'tight', 'format':'png'}

if os.environ['PATH'][1:6] == 'astro':
    _TOP_DIR = '/astro/store/phat/arlewis/'
else:
    _TOP_DIR = '/Users/alexialewis/research/PHAT/'


_DATA_DIR = os.path.join(_TOP_DIR, 'maps', 'analysis')
_WORK_DIR = os.path.join(_TOP_DIR, 'uvflux')
_PLOT_DIR = os.path.join(_WORK_DIR, 'plots')

CBARTICKSIZE = 12
AXISLABELSIZE = 18


fuvmodredfile = os.path.join(_DATA_DIR, 'mod_fuv_red.fits')#mod, atten. flux
fuvmodintfile = os.path.join(_DATA_DIR, 'mod_fuv_int.fits')
fuvobsfile = os.path.join(_DATA_DIR,'galex_fuv.fits')
fuvnobgfile = os.path.join(_DATA_DIR, 'galex_fuv_nobgsub.fits') 
fuvobsbgfile = os.path.join(_DATA_DIR, 'galex_fuv0.fits') #bg sub basic

nuvmodredfile = os.path.join(_DATA_DIR, 'mod_nuv_red.fits')#mod, atten. flux
nuvmodintfile = os.path.join(_DATA_DIR, 'mod_nuv_int.fits')
nuvobsfile = os.path.join(_DATA_DIR,'galex_nuv.fits')#obs, bg sub flux
nuvnobgfile = os.path.join(_DATA_DIR, 'galex_nuv_nobgsub.fits')
nuvobsbgfile = os.path.join(_DATA_DIR, 'galex_nuv0.fits') #bg sub basic

sfrfile = os.path.join(_DATA_DIR, 'sfr100.fits')
weightfile = os.path.join(_DATA_DIR, 'weights.fits')
avdavfile = os.path.join(_DATA_DIR, 'dust.fits') #cmd derived A_V + dA_V/2.
avfile = os.path.join(_DATA_DIR, 'dust_av.fits')
davfile = os.path.join(_DATA_DIR, 'dust_dav.fits')
anuvfile = os.path.join(_WORK_DIR, 'A_nuv.fits')
afuvfile = os.path.join(_WORK_DIR, 'A_fuv.fits')

w = pyfits.getdata(weightfile)
sel = (w == 0)
fuvfluxobs = pyfits.getdata(fuvobsfile) * w
fuvfluxobs_bg = pyfits.getdata(fuvobsbgfile) * w
fuvfluxobs_nobg = pyfits.getdata(fuvnobgfile) * w
fuvfluxmodint = pyfits.getdata(fuvmodintfile)
fuvfluxmodred = pyfits.getdata(fuvmodredfile)

nuvfluxobs = pyfits.getdata(nuvobsfile) * w
nuvfluxobs_bg = pyfits.getdata(nuvobsbgfile) * w
nuvfluxobs_nobg = pyfits.getdata(nuvnobgfile) * w
nuvfluxmodint = pyfits.getdata(nuvmodintfile)
nuvfluxmodred = pyfits.getdata(nuvmodredfile)

anuv = pyfits.getdata(anuvfile)
afuv = pyfits.getdata(afuvfile)

fuvfluxobs[sel] = np.nan
nuvfluxobs[sel] = np.nan

sfr100 = pyfits.getdata(sfrfile)
avdav = pyfits.getdata(avdavfile)
av = pyfits.getdata(avfile)
dav = pyfits.getdata(davfile)

fuvmagobs = astrogrid.flux.galex_flux2mag(fuvfluxobs, 'galex_fuv')
fuvmagmodint = astrogrid.flux.galex_flux2mag(fuvfluxmodint, 'galex_fuv')
fuvmagmodred = astrogrid.flux.galex_flux2mag(fuvfluxmodred, 'galex_fuv')
nuvmagobs = astrogrid.flux.galex_flux2mag(nuvfluxobs, 'galex_nuv')
nuvmagmodint = astrogrid.flux.galex_flux2mag(nuvfluxmodint, 'galex_nuv')
nuvmagmodred = astrogrid.flux.galex_flux2mag(nuvfluxmodred, 'galex_nuv')

cmap2 = plt.cm.Blues
cmap2.set_under('0.65')
cmap2.set_bad('0.65')

cmap3 = plt.cm.Purples
cmap3.set_under('0.65')
cmap3.set_bad('0.65')

cmap4 = plt.cm.Greens
cmap4.set_under('0.65')
cmap4.set_bad('0.65')

fuvfluxobslab = r'\textbf{$f_\mathrm{FUV}^\mathrm{obs}$}'
fuvfluxmodredlab = r'\textbf{$f_\mathrm{FUV}^\mathrm{SFH}$}'
fuvfluxmodintlab = r'\textbf{$f_\mathrm{FUV}^\mathrm{SFH,0}$}'

nuvfluxobslab = r'\textbf{$f_\mathrm{NUV}^\mathrm{obs}$}'
nuvfluxmodredlab = r'\textbf{$f_\mathrm{NUV}^\mathrm{SFH}$}'
nuvfluxmodintlab = r'\textbf{$f_\mathrm{NUV}^\mathrm{SFH,0}$}'
matchdustlab = r'$A_V + dA_V/2$'


matchdustlim = [-0.06, 1.45]

fuvdata = {'fluxobs':fuvfluxobs, 'fluxobs_nobg': fuvfluxobs_nobg,
           'fluxobs_bg':fuvfluxobs_bg, 'fluxmodint':fuvfluxmodint,
           'fluxmodred':fuvfluxmodred,
           'magobs':fuvmagobs, 'magmodint':fuvmagmodint,
           'magmodred':fuvmagmodred, 'ext':afuv}
nuvdata = {'fluxobs':nuvfluxobs, 'fluxobs_nobg': nuvfluxobs_nobg,
           'fluxobs_bg':nuvfluxobs_bg, 'fluxmodint':nuvfluxmodint,
           'fluxmodred':nuvfluxmodred,
           'magobs':nuvmagobs, 'magmodint':nuvmagmodint,
           'magmodred':nuvmagmodred, 'ext':anuv}


plot_laws1 = [('MW', '2.2', '1.0'), ('MW', '2.5', '1.0'),
              ('MW', '2.8', '1.0'), ('MW', '3.1', '1.0'),
              ('MW', '3.4', '1.0'), ('MW', '3.7', '1.0'),
              ('MW', '4.0', '1.0'), ('MW', '4.3', '1.0'),
              ('SMC', '3.1', '0.0')]
plotfile1 = os.path.join(_PLOT_DIR, 'mw_smc_dust_law.png')
make_plots(fuvdata, nuvdata, plot_laws1, plotfile=plotfile1, save=args.save)


plot_laws3 = [('Calz', '2.2', '0.0'), ('Calz', '2.5', '0.0'),
              ('Calz', '2.8', '0.0'), ('Calz', '3.1', '0.0'),
              ('Calz', '3.4', '0.0'), ('Calz', '3.7', '0.0'),
              ('Calz', '4.0', '0.0'), ('Calz', '4.3', '0.0')]
plotfile3 = os.path.join(_PLOT_DIR, 'calzetti_dust_law.png')
make_plots(fuvdata, nuvdata, plot_laws3, plotfile=plotfile3, save=args.save)


plot_laws4 = [('C10', '2.2', '0.0'), ('C10', '2.2', '0.2'),
              ('C10', '2.2', '0.4'), ('C10', '2.2', '0.6'),
              ('C10', '2.2', '0.8'), ('C10', '2.2', '1.0')]
plotfile4 = os.path.join(_PLOT_DIR, 'conroy_dust_law_2.2.png')
make_plots(fuvdata, nuvdata, plot_laws4, plotfile=plotfile4, save=args.save)


plot_laws5 = [('C10', '2.5', '0.0'), ('C10', '2.5', '0.2'),
              ('C10', '2.5', '0.4'), ('C10', '2.5', '0.6'),
              ('C10', '2.5', '0.8'), ('C10', '2.5', '1.0')]
plotfile5 = os.path.join(_PLOT_DIR, 'conroy_dust_law_2.5.png')
make_plots(fuvdata, nuvdata, plot_laws5, plotfile=plotfile5, save=args.save)


plot_laws6 = [('C10', '2.8', '0.0'), ('C10', '2.8', '0.2'),
              ('C10', '2.8', '0.4'), ('C10', '2.8', '0.6'),
              ('C10', '2.8', '0.8'), ('C10', '2.8', '1.0')]
plotfile6 = os.path.join(_PLOT_DIR, 'conroy_dust_law_2.8.png')
make_plots(fuvdata, nuvdata, plot_laws6, plotfile=plotfile6, save=args.save)


plot_laws7 = [('C10', '3.1', '0.0'), ('C10', '3.1', '0.2'),
              ('C10', '3.1', '0.4'), ('C10', '3.1', '0.6'),
              ('C10', '3.1', '0.8'), ('C10', '3.1', '1.0')]
plotfile7 = os.path.join(_PLOT_DIR, 'conroy_dust_law_3.1.png')
make_plots(fuvdata, nuvdata, plot_laws7, plotfile=plotfile7, save=args.save)


plot_laws8 = [('C10', '3.4', '0.0'), ('C10', '3.4', '0.2'),
              ('C10', '3.4', '0.4'), ('C10', '3.4', '0.6'),
              ('C10', '3.4', '0.8'), ('C10', '3.4', '1.0')]
plotfile8 = os.path.join(_PLOT_DIR, 'conroy_dust_law_3.4.png')
make_plots(fuvdata, nuvdata, plot_laws8, plotfile=plotfile8, save=args.save)


plot_laws9 = [('C10', '3.7', '0.0'), ('C10', '3.7', '0.2'),
              ('C10', '3.7', '0.4'), ('C10', '3.7', '0.6'),
              ('C10', '3.7', '0.8'), ('C10', '3.7', '1.0')]
plotfile9 = os.path.join(_PLOT_DIR, 'conroy_dust_law_3.7.png')
make_plots(fuvdata, nuvdata, plot_laws9, plotfile=plotfile9, save=args.save)


plot_laws10 = [('C10', '4.0', '0.0'), ('C10', '4.0', '0.2'),
               ('C10', '4.0', '0.4'), ('C10', '4.0', '0.6'),
               ('C10', '4.0', '0.8'), ('C10', '4.0', '1.0')]
plotfile10 = os.path.join(_PLOT_DIR, 'conroy_dust_law_4.0.png')
make_plots(fuvdata, nuvdata,plot_laws10,plotfile=plotfile10, save=args.save)


plot_laws11 = [('C10', '4.3', '0.0'), ('C10', '4.3', '0.2'),
               ('C10', '4.3', '0.4'), ('C10', '4.3', '0.6'),
               ('C10', '4.3', '0.8'), ('C10', '4.3', '1.0')]
plotfile11 = os.path.join(_PLOT_DIR, 'conroy_dust_law_4.3.png')
make_plots(fuvdata,nuvdata,plot_laws11, plotfile=plotfile11, save=args.save)
