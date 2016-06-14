import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter, LogFormatter
from matplotlib import rcParams
from scipy.stats import binned_statistic_2d
from pdb import set_trace
import compile_data
import seaborn as sns
import match_utils
import matplotlib.gridspec as gridspec
from sys import exit

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
    return parser.parse_args()



def set_colormaps():
    cmap = plt.cm.gist_heat_r
    cmap2, cmap3, cmap4 = plt.cm.Blues, plt.cm.Purples, plt.cm.Greens

    for cm in [cmap, cmap2, cmap3, cmap4]:
        cm.set_bad('0.75')
        #cm.set_under('0.75')
    return cmap, cmap2, cmap3, cmap4


def fig_uvmaps(uvdata, save=False):
    #labels = [fuvfluxobslab, fuvfluxmodredlab, fuvfluxmodintlab]
    labels = [r'\textbf{observed FUV}', r'\textbf{predicted FUV}']
    plotname = os.path.join(_PLOT_DIR,'fuv_maps_research_proposal.'+ plot_kwargs['format'])

    rcParams['axes.linewidth'] = 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,4))

    ax1.imshow(uvdata['fluxobs'][::-1].T, cmap=cmap, origin=origin,
               vmin=fluxmin, vmax=fluxmax, interpolation=interpolation)
    ax2.imshow(uvdata['fluxmodred'][::-1].T, cmap=cmap, origin=origin,
               vmin=fluxmin, vmax=fluxmax, interpolation=interpolation)

    for i, ax in enumerate([ax1, ax2]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.01, 0.89, labels[i], transform=ax.transAxes,
                fontsize=14)

    plt.subplots_adjust(hspace=0.0, left=0.05, right=0.95, bottom=0.05)
    if save:
        plt.savefig(plotname, **plot_kwargs)
    else:
        plt.show()
    rcParams['axes.linewidth'] = 2




sns.reset_orig()
args = get_args()
res = args.res
dust_curve = args.dust_curve

kwargs = {'density': args.density, 'outline': args.outline, 'save': args.save,
          'red': args.red, 'sfhs': args.sfhs}
plot_kwargs = {'dpi':300, 'bbox_inches':'tight', 'format':'png'}

if os.environ['PATH'][1:6] == 'astro':
    _TOP_DIR = '/astro/store/phat/arlewis/'
else:
    _TOP_DIR = '/Users/alexialewis/research/PHAT/'

_DATA_DIR = os.path.join(_TOP_DIR, 'maps', 'analysis')
_WORK_DIR = os.path.join(_TOP_DIR, 'uvflux')
_PLOT_DIR = '/Users/alexialewis/Dropbox/postdoc/11-02_jansky/'

CBARTICKSIZE, AXISLABELSIZE, PLOTTEXTSIZE = 12, 18, 16

sfr100lab = r'\textbf{$\langle\mathrm{SFR}\rangle_{100}$}'
fuvfluxobslab = r'\textbf{$f_\mathrm{FUV}^\mathrm{obs}$}'
fuvfluxmodredlab = r'\textbf{$f_\mathrm{FUV}^\mathrm{syn}$}'
fuvfluxmodintlab = r'\textbf{$f_\mathrm{FUV}^\mathrm{syn,0}$}'

nuvfluxobslab = r'\textbf{$f_\mathrm{NUV}^\mathrm{obs}$}'
nuvfluxmodredlab = r'\textbf{$f_\mathrm{NUV}^\mathrm{syn}$}'
nuvfluxmodintlab = r'\textbf{$f_\mathrm{NUV}^\mathrm{syn,0}$}'
matchdustlab = r'$A_V + dA_V/2$'

cmap, cmap2, cmap3, cmap4 = set_colormaps() #heat, blue, purple, green
origin = 'lower'
aspect = 'auto'
bins = 75
interpolation='none'

fluxmin, fluxmin2, fluxmax, fluxmaxint = 1e-20, 1e-16, 4e-15, 4e-14
matchdustlim = [-0.06, 1.52]
sfrmin, sfrmax = 1e-6, 3e-4

fuvdata, nuvdata, otherdata = compile_data.gather_map_data(res, dust_curve,
                                                           sfh=args.sfh)

## FUV & NUV maps ##
fig_uvmaps(fuvdata, save=args.save)
