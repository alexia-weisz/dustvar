import numpy as np
import matplotlib.pyplot as plt
from sedpy import attenuation, observate

import fsps


sps = fsps.StellarPopulation()
sps.params['sfh'] = 4
sps.params['const'] = 1.0
sps.params['imf_type'] = 2
wave, s = sps.get_spectrum(tage=1.0, peraa=True)

M31_DM = 24.47
ATT = attenuation.conroy

filters = ['galex_fuv', 'galex_nuv']
filters = observate.load_filters(filters)


def ext_func(rv, av, f_bump=1., att=attenuation.conroy):
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




if __name__ == '__main__':
    rvgrid = np.arange(0, 10, 0.1)
    fbgrid = np.arange(0, 2, 0.05)

    ebv = []
    for rv in rvgrid:
        for fb in fbgrid:
            model = ext_func(rv, 1.0, f_bump=fb)
            ebv.append(model[-1])

