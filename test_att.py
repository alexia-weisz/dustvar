import numpy as np
import fsps

import matplotlib.pyplot as plt

from sedpy import attenuation, observate


def create_spectrum(tage=1.0):
    """
    Create a spectrum at a given age via fsps.StelalarPopulation().get_spectrum
    """
    sps = fsps.StellarPopulation()
    sps.params['sfh'] = 4
    sps.params['const'] = 1.0
    sps.params['imf_type'] = 2

    wave, s = sps.get_spectrum(tage=tage, peraa=True)
    return wave, s

filters = ['galex_fuv', 'galex_nuv']
filters = observate.load_filters(filters)

wave, s = create_spectrum(tage=0.2)
wave_micron = wave / 1e4
fig, ax = plt.subplots(2, 2, figsize=(8,5), sharey=True, sharex=True)

lw = 0.2
color = 'gray'

rv1 = np.arange(3, 3.5, 0.01)
rv2 = np.arange(3, 3.1, 0.001)

for r in rv1:
    tau_lambda_cardelli = attenuation.cardelli(wave, R_v=r, tau_v=1.0)
    A_lambda_cardelli = np.log10(np.exp(1)) * tau_lambda_cardelli

    tau_lambda_conroy = attenuation.conroy(wave, R_v=r, f_bump=1.0, tau_v=1.0)
    A_lambda_conroy = np.log10(np.exp(1)) * tau_lambda_conroy

    sel = (wave > 5489) & (wave < 5511)
    A_V_cardelli = np.mean(A_lambda_cardelli[sel])
    A_V_conroy = np.mean(A_lambda_conroy[sel])

    ax[0,0].plot(1./wave_micron, A_lambda_cardelli/A_V_cardelli, lw=lw,
             color=color, ls='-', label='Cardelli', alpha=0.3)
    ax[0,1].plot(1./wave_micron, A_lambda_conroy/A_V_conroy, lw=lw,
             color=color, ls='-', label='Conroy', alpha=0.3)

for r in rv2:
    tau_lambda_cardelli = attenuation.cardelli(wave, R_v=r, tau_v=1.0)
    A_lambda_cardelli = np.log10(np.exp(1)) * tau_lambda_cardelli

    tau_lambda_conroy = attenuation.conroy(wave, R_v=r, f_bump=1.0, tau_v=1.0)
    A_lambda_conroy = np.log10(np.exp(1)) * tau_lambda_conroy

    sel = (wave > 5489) & (wave < 5511)
    A_V_cardelli = np.mean(A_lambda_cardelli[sel])
    A_V_conroy = np.mean(A_lambda_conroy[sel])

    ax[1,0].plot(1./wave_micron, A_lambda_cardelli/A_V_cardelli, lw=lw,
             color=color, ls='-', label='Cardelli', alpha=0.3)
    ax[1,1].plot(1./wave_micron, A_lambda_conroy/A_V_conroy, lw=lw,
             color=color, ls='-', label='Conroy', alpha=0.3)

tx, ty = 0.75, 0.1
ax[0,0].text(tx, ty, 'Cardelli', transform=ax[0,0].transAxes)
ax[0,0].text(0.05, 0.8, '$R_V=[3, 3.5, 0.01]$', transform=ax[0,0].transAxes)
ax[1,0].text(tx, ty, 'Cardelli', transform=ax[1,0].transAxes)
ax[1,0].text(0.05, 0.8, '$R_V=[3, 3.1, 0.001]$', transform=ax[1,0].transAxes)
ax[0,1].text(tx, ty, 'Conroy', transform=ax[0,1].transAxes)
ax[1,1].text(tx, ty, 'Conroy', transform=ax[1,1].transAxes)


ax[1,0].set_ylabel(r'$A_\lambda / A_V$', fontsize=22)
ax[1,0].set_xlabel(r'1/$\lambda$ [$\mu \textrm{m}^{-1}$]', fontsize=22)
#ax2.set_xlabel(r'1/$\lambda$ [$\mu \textrm{m}^{-1}$]', fontsize=22)
ax[0,0].set_xlim(0.2, 8.2)
ax[0,0].set_ylim(0, 4)
ax[0,0].tick_params(axis='both', labelsize=18)
#ax2.tick_params(axis='both', labelsize=18)

plt.show()
