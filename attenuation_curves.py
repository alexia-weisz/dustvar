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

def get_alambda(dust_curve, wave, R_v=None, f_bump=None, tau_v=None, get_av=True):
    kwargs = {}
    if R_v is not None:
        kwargs['R_v'] = R_v
    if f_bump is not None:
        kwargs['f_bump'] = f_bump
    if tau_v is not None:
        kwargs['tau_v'] = tau_v
    tau_lambda = dust_curve(wave, **kwargs)
    A_lambda = np.log10(np.exp(1)) * tau_lambda

    if get_av:
        sel = (wave > 5489) & (wave < 5511)
        A_V = np.mean(A_lambda[sel])
        return A_lambda, A_V

    return A_lambda


def fig_att_curves(tage=0.0, **kwargs):

    filters = ['galex_fuv', 'galex_nuv', 'wfc3_uvis_f475w', 'wfc3_uvis_f814w']
    filters = observate.load_filters(filters)

    wave, s = create_spectrum(tage=0.2)
    A_lambda_calzetti, A_V_calzetti = get_alambda(attenuation.calzetti, wave, R_v=4.05, tau_v=1.0)
    A_lambda_cardelli, A_V_cardelli = get_alambda(attenuation.cardelli, wave, R_v=3.10, tau_v=1.0)
    A_lambda_smc, A_V_smc = get_alambda(attenuation.smc, wave, tau_v=1.0)

    wave_micron = wave / 1e4

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
    ax.patch.set_visible(False)

    lw = 5
    color = 'black'
    zorder = 100
    ax.plot(1./wave_micron, A_lambda_cardelli/A_V_cardelli, lw=lw,
             color=color, ls='-', label='Cardelli', zorder=zorder)
    ax.plot(1./wave_micron, A_lambda_calzetti/A_V_calzetti, lw=lw,
             color=color, ls='--', label='Calzetti', zorder=zorder)
    ax.plot(1./wave_micron, A_lambda_smc/A_V_smc, lw=lw,
             color=color, ls=':', label='SMC', zorder=zorder)

    textx = [0.78, 0.48, 0.22, 0.07]
    texty = 0.93
    colors = ['indigo', 'darkviolet', 'blue', 'red']
    #cc = [color['color'] for color in list(plt.rcParams['axes.prop_cycle'])]
    #colors = [cc[5], cc[0], cc[4], cc[7]]
    labels = [r'\textbf{FUV}', r'\textbf{NUV}', r'\textbf{F475W}', r'\textbf{F814W}']
    for i in range(len(filters)):
        xx = 1./(filters[i].wavelength/1e4)
        yy = filters[i].transmission
        ax2.plot(xx, yy/np.max(yy), lw=0)#, color=colors[i], label=labels[i])
        ax2.fill_between(xx, 0, yy/np.max(yy), lw=0, edgecolor=colors[i],
                         facecolor=colors[i], label=labels[i], alpha=0.2)
        ax2.text(textx[i], texty, labels[i], color=colors[i], fontsize=14, transform=ax2.transAxes)

    ax.set_ylabel(r'$A_\lambda / A_V$', fontsize=22)
    ax.set_xlabel(r'1/$\lambda$ [$\mu \textrm{m}^{-1}$]', fontsize=22)
    ax2.set_ylabel('Filter Transmission', fontsize=22, labelpad=15)
    ax.set_xlim(0.2, 8.2)
    ax.set_ylim(0, 8)
    ax2.set_ylim(0.05, 1.1)

    for a in [ax, ax2]:
        a.tick_params(axis='both', labelsize=18)


    plotname = 'attenuation_curves.pdf'
    if kwargs['save']:
        plt.savefig(plotname, **plot_kwargs)
    else:
        plt.show()
