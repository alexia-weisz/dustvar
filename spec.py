def spectrum(sfr, age, **kwargs):
    if len(age) == 2:
        try:
            age = np.append(age[0], age[1][-1])  # One array of bin edges
        except (TypeError, IndexError):
            # Probably not a length-2 sequence of sequences
            pass

    age_list = kwargs.get('age_observe', 0.0001)
    try:
        len_age_list = len(age_list)
    except TypeError:
        # age_list is a single value
        age_list = [age_list]
        len_age_list = 0

    bin_res = kwargs.get('bin_res', 20.0)
    av, dav = kwargs.get('av', None), kwargs.get('dav', None)
    rv, f_bump = kwargs.get('rv', 3.1), kwargs.get('f_bump', 1.0)
    nsplit = kwargs.get('nsplit', 30)
    dust_curve = kwargs.get('dust_curve', 'cardelli')
    if isinstance(dust_curve, basestring):
        dust_curve = DUST_CURVE[dust_curve]
    fsps_kwargs = kwargs.get('fsps_kwargs', {})

    # To save time, create StellarPopulation only when necessary
    try:
        sps = CURRENT_SP[0]
    except IndexError:
        sps = fsps.StellarPopulation()
        CURRENT_SP.append(sp)
    fsps_kwargs['sfh'] = 0
    for key, val in fsps_kwargs.items():
        sps.params[key] = val

    names = ['t1', 't2', 'sfr']
    names = [name.encode('utf-8') for name in names]# unicode names not allowed
    types = [float, float, float]
    dtypes = zip(names, types)
    sfh = np.array(zip(age[:-1], age[1:], sfr), dtypes)
    # Resample the SFH to a high time resolution
    age, sfr = bursty_sfh.burst_sfh(f_burst=0, sfh=sfh, bin_res=bin_res)[:2]



    lt = age
    lookback_time=age_list

    sps.params['sfh'] = 0  # make sure SSPs
    ssp_ages = 10**sps.ssp_ages  # in yrs
    if logzsol is None:
        wave, spec = sps.get_spectrum(peraa=True, tage=0)
        mass = sps.stellar_mass.copy()
    else:
        assert(sps._zcontinuous > 0)
        spec, mass = [], []
        for tage, logz in zip(ssp_ages/1e9, logzsol):
            sps.params['logzsol'] = logz
            spec.append(sps.get_spectrum(peraa=True, tage=tage)[1])
            mass.append(sps.stellar_mass)
        spec = np.array(spec)
        mass = np.array(mass)
        wave = sps.wavelengths


    # Redden the SSP spectra
    spec, lir = redden(wave, spec, rv=rv, f_bump=f_bump, av=av, dav=dav,
                       dust_curve=dust_curve, nsplit=nsplit)

    # Get interpolation weights based on the SFH
    target_lt = np.atleast_1d(lookback_time)
    aw = bursty_sfh.sfh_weights(lt, sfr, ssp_ages, lookback_time=target_lt, **extras)

    # Do the linear combination
    int_spec = (spec[None,:,:] * aw[:,:,None]).sum(axis=1)
    mstar = (mass[None,:] * aw).sum(axis=-1)
    if lir is not None:
        lir_tot = (lir[None,:] * aw).sum(axis = -1)
    else:
        lir_tot = 0

    output = wave, int_spec, mstar, lir_tot



    wave, spec, mstar, lum_ir = output

    if not len_age_list:
        spec, mstar, lum_ir = spec[0], mstar, lum_ir

    return wave, spec, lum_ir
