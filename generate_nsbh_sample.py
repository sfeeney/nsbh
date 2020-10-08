import numpy as np
import numpy.random as npr
import scipy.stats as ss
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import scipy.interpolate as si
import os
import xml.etree.ElementTree as et
import bilby
import bilby.gw.utils as bu
import bilby.gw.conversion as bc
import bilby.gw.detector as bd
import lalsimulation as lalsim
import astropy.time as at
import astropy.coordinates as ac
import astropy.units as au
import ns_eos_aw as nseos
import math
import pickle
import getdist as gd
import getdist.plots as gdp

def dd2_lambda_from_mass(m):
    return 1.60491e6 - 23020.6 * m**-5 + 194720. * m**-4 - 658596. * m**-3 \
        + 1.33938e6 * m**-2 - 1.78004e6 * m**-1 - 992989. * m + 416080. * m**2 \
        - 112946. * m**3 + 17928.5 * m**4 - 1263.34 * m**5

def inverse_transform_precessing_spins(iota, spin_1x, spin_1y, \
                                       spin_1z, spin_2x, spin_2y, \
                                       spin_2z, mass_1, mass_2, \
                                       reference_frequency, phase):

    args_list = bu.convert_args_list_to_float(
        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, 
        mass_1, mass_2, reference_frequency, phase)
    results = lalsim.SimInspiralTransformPrecessingWvf2PE(*args_list)
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = (results)

    return theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2

def d2z(d, h_0, q_0, order=3):

    z = h_0 * d / c
    if order > 1:
        z += -1.0 / 2.0 * (1.0 - q_0) * (h_0 * d / c) ** 2
    if order > 2:
        z += 1.0 / 6.0 * (4.0 - 7.0 * q_0 + 1.0) * (h_0 * d / c) ** 3

    return z

def z2d(z, h_0, q_0, order=3):

    d = c * z / h_0
    if order > 1:
        d += 1.0 / 2.0 * (1.0 - q_0) * c * z ** 2 / h_0
    if order > 2:
        d += 1.0 / 6.0 * (-1.0 + q_0 - 1.0 + 3.0 * q_0 ** 2) * \
             c * z ** 3 / h_0

    return d

def dz_dd(d, h_0, q_0, order=3):

    dzdd = h_0 / c
    if order > 1:
        dzdd += -1.0 * (1.0 - q_0) * h_0 * d / c
    if order > 2:
        dzdd += 1.0 / 2.0 * (4.0 - 7.0 * q_0 + 1.0) * (h_0 * d / c) ** 2

    return dzdd

def dvolume_dd(d, h_0, q_0, order=3):

    dvdd = 4.0 * np.pi * d ** 2
    if order > 1:
        dvdd += -16.0 * np.pi * h_0 / c * d ** 3
    if order > 2:
        dvdd += 2.0 * np.pi * (25.0 - 5.0 * q_0 + 4.0 * q_0 ** 2) * \
                 (h_0 / c) ** 2 * d ** 4

    return dvdd

# volume in Mpc^3
def volume(d, d_min, h_0, q_0, order=3):

    vol = 4.0 * np.pi * (d ** 3 - d_min ** 3) / 3.0
    if order > 1:
        vol += -4.0 * np.pi * (d ** 4 - d_min ** 4) * h_0 / c
    if order > 2:
        vol += 4.0 * np.pi * (d ** 5 - d_min ** 5) / 10.0 * \
               (25.0 - 5.0 * q_0 + 4.0 * q_0 ** 2) * (h_0 / c) ** 2
    return vol

def dvolume_dz(z, h_0, q_0, order=3, redshift_rate=False):

    dvdz = 1.0
    if order > 1:
        dvdz += -2.0 * (1.0 + q_0) * z
    if order > 2:
        dvdz += 5.0 / 12.0 * (7.0  + 14.0 * q_0 - 2.0 + 9 * q_0 ** 2) * \
                z ** 2
    dvdz *= 4.0 * np.pi * (c / h_0) ** 3 * z ** 2
    if redshift_rate:
        dvdz /= (1.0 + z)

    return dvdz

def volume_z(z, z_min, h_0, q_0, order=3, redshift_rate=False):

    if redshift_rate:

        vol = 1.0 / 2.0 * (z ** 2 - z_min ** 2) - \
              (z - z_min) + np.log((1 + z) / (1 + z_min))
        if order > 1:
            vol += -2.0 * (1.0 + q_0) * \
                   (1.0 / 3.0 * (z ** 3 - z_min ** 3) - \
                    1.0 / 2.0 * (z ** 2 - z_min ** 2) + \
                    (z - z_min) - np.log((1 + z) / (1 + z_min)))
        if order > 2:
            vol += 5.0 / 12.0 * (7.0  + 14.0 * q_0 - 2.0 + 9 * q_0 ** 2) * \
                   (1.0 / 4.0 * (z ** 4 - z_min ** 4) - \
                    1.0 / 3.0 * (z ** 3 - z_min ** 3) + \
                    1.0 / 2.0 * (z ** 2 - z_min ** 2) - \
                    (z - z_min) + np.log((1 + z) / (1 + z_min)))
        vol *= 4.0 * np.pi * (c / h_0) ** 3

    else:

        vol = 1.0 / 3.0 * (z ** 3 - z_min ** 3)
        if order > 1:
            vol += -1.0 / 2.0 * (1.0 + q_0) * (z ** 4 - z_min ** 4)
        if order > 2:
            vol += 1.0 / 12.0 * (7.0  + 14.0 * q_0 - 2.0 + 9 * q_0 ** 2) * \
                   (z ** 5 - z_min ** 5)
        vol *= 4.0 * np.pi * (c / h_0) ** 3

    return vol

def nsbh_population(rate, t_min, t_max, f_online, d_min, d_max, h_0,\
                    q_0, m_min_1, m_max_1, m_mean_1, m_std_1, m_min_2, \
                    m_max_2, m_mean_2, m_std_2, a_min_1, a_max_1, \
                    a_min_2, a_max_2, seed=None, sample_z=False, \
                    redshift_rate=False, uniform_bh_masses=False, \
                    uniform_ns_masses=False, fixed_count=None, \
                    aligned_spins=False):

    # constrained realisation if desired
    if seed is not None:
        npr.seed(seed)

    # first draw number of events: a Poisson process, with rate 
    # given by the number of events per year per Gpc^3, the 
    # duration of observations and the volume
    if fixed_count is None:
        if sample_z:
            z_min = d2z(d_min, h_0, q_0)
            z_max = d2z(d_max, h_0, q_0)
            vol = volume_z(z_max, z_min, h_0, q_0, \
                           redshift_rate=redshift_rate)
        else:
            vol = volume(d_max, d_min, h_0, q_0)
        n_per_sec = rate * vol / 365.0 / 24.0 / 3600.0 * f_online
        n_exp = n_per_sec * (t_max - t_min)
        n_inj = npr.poisson(n_exp)
    else:
        if sample_z:
            z_min = d2z(d_min, h_0, q_0)
            z_max = d2z(d_max, h_0, q_0)
        n_inj = fixed_count
        n_per_sec = fixed_count / (t_max - t_min)

    # draw merger times consistent with the expected rate. add a 
    # check to ensure that the latest merger time is within the 
    # observation window
    times = np.zeros(n_inj)
    times[0] = t_min
    times[-1] = t_max + 1.0
    while times[-1] >= t_max:
        delta_times = npr.exponential(1.0 / n_per_sec, n_inj - 1)
        for i in range(1, n_inj):
            times[i] = times[i - 1] + delta_times[i - 1]

    # draw distances via an interpolated CDF
    if sample_z:

        z_grid = np.linspace(z_min, z_max, 10000)
        p_z_grid = volume_z(z_grid, z_min, h_0, q_0, \
                            redshift_rate=redshift_rate) / \
                   volume_z(z_max, z_min, h_0, q_0, \
                            redshift_rate=redshift_rate)
        interp = si.interp1d(p_z_grid, z_grid)
        redshifts = interp(npr.uniform(size=n_inj))
        distances = z2d(redshifts, h_0, q_0)

    else:

        d_grid = np.linspace(d_min, d_max, 10000)
        p_d_grid = volume(d_grid, d_min, h_0, q_0) / \
                   volume(d_max, d_min, h_0, q_0)
        interp = si.interp1d(p_d_grid, d_grid)
        distances = interp(npr.uniform(size=n_inj))

    # draw inclinations, colatitudes and longitudes
    incs = np.arccos(-npr.uniform(-1.0, 1.0, size=n_inj))
    colats = np.arcsin(-npr.uniform(-1.0, 1.0, size=n_inj))
    longs = npr.uniform(0.0, 2.0 * np.pi, size=n_inj)

    # draw masses
    if uniform_bh_masses:
        m_1s = npr.uniform(m_min_1, m_max_1, size=n_inj)
    else:
        dist = ss.truncnorm((m_min_1 - m_mean_1) / m_std_1, \
                            (m_max_1 - m_mean_1) / m_std_1, \
                            loc=m_mean_1, scale=m_std_1)
        m_1s = dist.rvs(n_inj)
    if uniform_ns_masses:
        m_2s = npr.uniform(m_min_2, m_max_2, size=n_inj)
    else:
        dist = ss.truncnorm((m_min_2 - m_mean_2) / m_std_2, \
                            (m_max_2 - m_mean_2) / m_std_2, \
                            loc=m_mean_2, scale=m_std_2)
        m_2s = dist.rvs(n_inj)

    # now draw spins: isotropic in direction, uniform in magnitude
    spin_amps = npr.uniform(a_min_1, a_max_1, size=n_inj)
    spin_colats = np.arccos(-npr.uniform(-1.0, 1.0, size=n_inj))
    spin_longs = npr.uniform(0.0, 2.0 * np.pi, size=n_inj)
    a_1_xs = spin_amps * np.sin(spin_colats) * np.cos(spin_longs)
    a_1_ys = spin_amps * np.sin(spin_colats) * np.sin(spin_longs)
    a_1_zs = spin_amps * np.cos(spin_colats)
    if aligned_spins:
        a_1_xs = 0.0
        a_1_ys = 0.0
    spin_amps = npr.uniform(a_min_2, a_max_2, size=n_inj)
    spin_colats = np.arccos(-npr.uniform(-1.0, 1.0, size=n_inj))
    spin_longs = npr.uniform(0.0, 2.0 * np.pi, size=n_inj)
    a_2_xs = spin_amps * np.sin(spin_colats) * np.cos(spin_longs)
    a_2_ys = spin_amps * np.sin(spin_colats) * np.sin(spin_longs)
    a_2_zs = spin_amps * np.cos(spin_colats)
    if aligned_spins:
        a_2_xs = 0.0
        a_2_ys = 0.0

    # finally draw isotropic coa_phase and polarization angles
    coa_phases = npr.uniform(0.0, 2.0 * np.pi, size=n_inj)
    pols = npr.uniform(0.0, 2.0 * np.pi, size=n_inj)

    # store in structured array
    dtypes = [('simulation_id', 'U256'), ('mass1', float), \
              ('mass2', float), ('spin1x', float), ('spin1y', float), \
              ('spin1z', float), ('spin2x', float), \
              ('spin2y', float), ('spin2z', float), ('redshift', float), \
              ('distance', float), ('inclination', float), \
              ('coa_phase', float), ('polarization', float), \
              ('longitude', float), ('latitude', float), \
              ('geocent_end_time', int), ('geocent_end_time_ns', int)]
    data = np.empty((n_inj, ), dtype=dtypes)
    data['simulation_id'] = \
        ['sim_inspiral:simulation_id:{:d}'.format(i) for i in range(n_inj)]
    data['mass1'] = m_1s
    data['mass2'] = m_2s
    data['spin1x'] = a_1_xs
    data['spin1y'] = a_1_ys
    data['spin1z'] = a_1_zs
    data['spin2x'] = a_2_xs
    data['spin2y'] = a_2_ys
    data['spin2z'] = a_2_zs
    data['redshift'] = redshifts
    data['distance'] = distances
    data['inclination'] = incs
    data['coa_phase'] = coa_phases
    data['polarization'] = pols
    data['longitude'] = longs
    data['latitude'] = colats
    data['geocent_end_time'] = [int(math.modf(t)[1]) for t in times]
    data['geocent_end_time_ns'] = [int(math.modf(t)[0] * 1e9) for t in times]

    return 1.0 / n_per_sec, data

def comp_masses_to_chirp_q(m_1, m_2):

    m_c = (m_1 * m_2) ** 0.6 / (m_1 + m_2) ** 0.2
    q_inv = m_2 / m_1

    return m_c, q_inv

def chirp_q_to_comp_masses(m_c, q_inv):

    q = 1.0 / q_inv
    m_2 = (1 + q) ** 0.2 / q ** 0.6 * m_c
    m_1 = q * m_2

    return m_1, m_2


# plot settings
lw = 1.5
mp.rc('font', family='serif', size=10)
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# settings
min_network = False
if min_network:
    ifo_list = ['H1', 'L1', 'V1', 'K1-']
    t_obs = 3.0 # years
    d_max = 1500.0 # Mpc
else:
    ifo_list = ['H1+', 'L1+', 'V1+', 'K1+', 'A1']
    t_obs = 5.0 # years
    d_max = 2045.0 # Mpc # I used 2250.0!
c = 2.998e5 # km / s
h_0 = 67.36 # km / s / Mpc
q_0 = 0.5 * 0.3153 - 0.6847
rate = 6.1e-7 # 2.0e-6 # events / year / Mpc^3
d_min = 0.0
t_start = 1325030418
t_stop = t_start + t_obs * 3600 * 24 * 365
f_online = 0.5
seed = 141023
to_store = ['simulation_id', 'mass1', 'mass2', 'spin1x', 'spin1y', \
            'spin1z', 'spin2x' , 'spin2y', 'spin2z', 'redshift', \
            'distance', 'inclination', 'coa_phase', 'polarization', \
            'longitude', 'latitude', 'geocent_end_time', \
            'geocent_end_time_ns']
n_to_store = len(to_store)
use_lal = False
sample_z = True
redshift_rate = True
uniform_bh_masses = True
uniform_ns_masses = True
low_metals = True
broad_bh_spins = True
seobnr_waveform = False
if seobnr_waveform:
    waveform_approximant = 'SEOBNRv4_ROM_NRTidalv2_NSBH'
    aligned_spins = True
else:
    waveform_approximant = 'IMRPhenomPv2_NRTidal'
    aligned_spins = False
cf_bilby = False

# BH mass and spin dists
if uniform_bh_masses:
    m_min_bh = 2.5
    if low_metals:
        m_max_bh = 40.0
    else:
        m_max_bh = 12.0
else:
    m_min_bh = 5.0
    m_max_bh = 20.0
m_mean_bh = 8.0
m_std_bh = 1.0
spin_min_bh = 0.0
if broad_bh_spins:
    spin_max_bh = 0.99
else:
    spin_max_bh = 0.5

# NS mass and spin dists
m_min_ns = 1.0
if uniform_ns_masses:
    m_max_ns = 2.42
else:
    m_max_ns = 2.0
m_mean_ns = 1.33
m_std_ns = 0.15
spin_min_ns = 0.0
spin_max_ns = 0.05

# Bilby settings
snr_thresh = 12.0
duration = 32.0
sampling_frequency = 2048.
minimum_frequency = 20.0
reference_frequency = 14.0
outdir = 'data'

# filename stub
if ifo_list == ['H1', 'L1', 'V1']:
    ifo_str = ''
else:
    ifo_str = '_'.join(ifo_list) + '_'
label_str = 'nsbh_pop_' + ifo_str + \
            'd_{:04.1f}_mf_{:4.1f}_rf_{:4.1f}'
if sample_z:
    label_str += '_dndz'
    if redshift_rate:
        label_str += '_rr'
if uniform_bh_masses:
    label_str += '_ubhmp_{:.1f}_{:.1f}'.format(m_min_bh, m_max_bh)
if uniform_ns_masses:
    label_str += '_unsmp_{:.1f}_{:.1f}'.format(m_min_ns, m_max_ns)
if broad_bh_spins:
    label_str += '_bbhsp'
if seobnr_waveform:
    label_str += '_seobnr'
if aligned_spins:
    label_str += '_aligned'
label = label_str.format(duration, minimum_frequency, \
                         reference_frequency)

# generate raw simulation using LAL or my version
if use_lal:

    # rate calculations
    volume = 4.0 * np.pi * d_max ** 3 * (1.0 / 3.0 - h_0 * d_max / c)
    events_per_year = rate * volume
    s_per_event = 365.0 * 24.0 * 3600.0 / events_per_year

    # build up and execute command-line LAL call
    print('sampling population priors')
    xml_file = 'data/' + label + '_raw_pars.xml'
    lal_cmd = 'lalapps_inspinj --t-distr exponential ' + \
              '--time-step {:.1f} '.format(s_per_event) + \
              '--gps-start-time {:d} '.format(t_start) + \
              '--gps-end-time {:d} '.format(t_stop) + \
              '--l-distr random --waveform ' + waveform_approximant + ' ' + \
              '--i-distr uniform --max-inc 90 --d-distr volume ' + \
              '--min-distance 1 --max-distance {:.1f} '.format(d_max * \
                                                               1000.0) + \
              '--m-distr gaussian --min-mass1 {:.1f} '.format(m_min_bh) + \
              '--max-mass1 {:.1f} '.format(m_max_bh) + \
              '--mean-mass1 {:.1f} '.format(m_mean_bh) + \
              '--stdev-mass1 {:.1f} '.format(m_std_bh) + \
              '--min-mass2 {:.1f} '.format(m_min_ns) + \
              '--max-mass2 {:.1f} '.format(m_max_ns) + \
              '--mean-mass2 {:.2f} '.format(m_mean_ns) + \
              '--stdev-mass2 {:.2f} '.format(m_std_ns) + \
              '--min-mtotal {:.1f} '.format(m_min_bh + m_min_ns) + \
              '--max-mtotal {:.1f} '.format(m_max_bh + m_max_ns) + \
              '--enable-spin --min-spin1 {:.1f} '.format(spin_min_bh) + \
              '--max-spin1 {:.1f} '.format(spin_max_bh) + \
              '--min-spin2 {:.1f} '.format(spin_min_ns) + \
              '--max-spin2 {:.2f} '.format(spin_max_ns) + \
              '--f-lower 9.0 --disable-milkyway ' + \
              '--output ' + xml_file + \
              ' --taper start --seed {:d}'.format(seed)
    os.system(lal_cmd)

    # parse into a sane format. find the data table. figure out the 
    # indices of the columns we want to store.
    tree = et.parse(xml_file)
    root = tree.getroot()
    data_table = root.find("./Table[@Name='sim_inspiral:table']")
    columns = [col.get('Name') for col in data_table.iter('Column')]
    types = [col.get('Type') for col in data_table.iter('Column')]
    ind_to_store = [columns.index(col) for col in to_store]
    types_to_store = [types[ind] for ind in ind_to_store]
    raw_data = data_table.find('Stream').text.split(',\n')

    # numpy dtype shenanigans
    dtypes = []
    for j in range(n_to_store):
        if types_to_store[j] == 'ilwd:char':
            dtypes.append((to_store[j], 'U256'))
        elif types_to_store[j] == 'real_4':
            dtypes.append((to_store[j], float))
        elif types_to_store[j] == 'int_4s':
            dtypes.append((to_store[j], int))

    # extract the required data
    n_inj = (len(raw_data))
    data = []
    for i in range(n_inj):
        inj_list = raw_data[i].split(',')
        inj_list = [inj_list[ind].strip() for ind in ind_to_store]
        for j in range(n_to_store):
            if types_to_store[j] == 'ilwd:char':
                inj_list[j] = inj_list[j].replace('"', '')
            elif types_to_store[j] == 'real_4':
                inj_list[j] = float(inj_list[j])
            elif types_to_store[j] == 'int_4s':
                inj_list[j] = int(inj_list[j])
        data.append(tuple(inj_list))
    data = np.array(data, dtype=dtypes)

else:

    # option to compare my priors to bilby's
    if not cf_bilby:

        # simulate using my code
        pop = nsbh_population(rate, t_start, t_stop, f_online, d_min, \
                              d_max, h_0, q_0, m_min_bh, m_max_bh, \
                              m_mean_bh, m_std_bh, m_min_ns, m_max_ns, \
                              m_mean_ns, m_std_ns, spin_min_bh, \
                              spin_max_bh, spin_min_ns, spin_max_ns, \
                              seed=seed, sample_z=sample_z, \
                              redshift_rate=redshift_rate, \
                              uniform_bh_masses=uniform_bh_masses, \
                              uniform_ns_masses=uniform_ns_masses, \
                              aligned_spins=aligned_spins)
        s_per_event = pop[0]
        data = pop[1]
        n_inj = data.shape[0]

    else:

        # simulate using my code
        pop = nsbh_population(rate, t_start, t_stop, f_online, d_min, \
                              d_max, h_0, q_0, m_min_bh, m_max_bh, \
                              m_mean_bh, m_std_bh, m_min_ns, m_max_ns, \
                              m_mean_ns, m_std_ns, spin_min_bh, \
                              spin_max_bh, spin_min_ns, spin_max_ns, \
                              seed=seed, sample_z=sample_z, \
                              redshift_rate=redshift_rate, \
                              uniform_bh_masses=uniform_bh_masses, \
                              uniform_ns_masses=uniform_ns_masses, \
                              fixed_count=50000, \
                              aligned_spins=aligned_spins)
        data = pop[1]
        n_inj = data.shape[0]

        # calculate limits on bilby priors
        if sample_z:
            d_min = z2d(d2z(d_min, h_0, q_0), h_0, q_0)
            d_max = z2d(d2z(d_max, h_0, q_0), h_0, q_0)
        m_c_min, _ = comp_masses_to_chirp_q(m_min_bh, m_min_ns)
        m_c_max, _ = comp_masses_to_chirp_q(m_max_bh, m_max_ns)
        _, q_inv_min = comp_masses_to_chirp_q(m_max_bh, m_min_ns)
        _, q_inv_max = comp_masses_to_chirp_q(m_min_bh, m_max_ns)

        # sample from bilby NSBH prior by modifying default prior for BNS
        priors = bilby.gw.prior.BNSPriorDict(aligned_spin=False)
        priors.pop('mass_1')
        priors.pop('mass_2')
        priors['chirp_mass'] = bilby.prior.Uniform(name='chirp_mass', \
                                                   unit='$M_{\\odot}$', \
                                                   latex_label='$M$', \
                                                   minimum=m_c_min, \
                                                   maximum=m_c_max)
        priors['mass_ratio'] = bilby.prior.Uniform(name='mass_ratio', \
                                                   latex_label='$q$', \
                                                   minimum=q_inv_min, \
                                                   maximum=q_inv_max)
        priors['geocent_time'] = \
            bilby.core.prior.Uniform(minimum=-0.1, maximum=0.0, \
                                     name='geocent_time', \
                                     latex_label='$t_c$', \
                                     unit='$s$')
        priors['lambda_2'] = bilby.core.prior.Uniform(name='lambda_2', \
                                                      minimum=0.0, \
                                                      maximum=4000.0, \
                                                      latex_label=r'$\Lambda_2$', \
                                                      boundary=None)
        priors.pop('a_1')
        priors['a_1'] = \
            bilby.core.prior.Uniform(name='a_1', minimum=spin_min_bh, \
                                     maximum=spin_max_bh, \
                                     boundary='reflective')
        priors.pop('luminosity_distance')
        priors['luminosity_distance'] = \
            bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', \
                                              minimum=d_min, maximum=d_max, \
                                              unit='Mpc', boundary=None)
        priors['lambda_1'] = 0.0
        bilby_samples = priors.sample(50000)

        # convert samples as required
        smf_m_c, smf_q_inv = \
            comp_masses_to_chirp_q(data['mass1'], data['mass2'])
        smf_samples = np.array([data['distance'], \
                                data['mass1'], \
                                data['mass2'], \
                                smf_m_c, smf_q_inv]).T
        bilby_m_bh, bilby_m_ns = \
            chirp_q_to_comp_masses(bilby_samples['chirp_mass'], \
                                   bilby_samples['mass_ratio'])
        bgw_samples = np.array([bilby_samples['luminosity_distance'], \
                                bilby_m_bh, \
                                bilby_m_ns, \
                                bilby_samples['chirp_mass'], \
                                bilby_samples['mass_ratio']]).T
        _, m_min_ns_bgw = chirp_q_to_comp_masses(m_c_min, q_inv_min)
        _, m_max_ns_bgw = chirp_q_to_comp_masses(m_c_max, q_inv_max)
        m_min_bh_bgw, _ = chirp_q_to_comp_masses(m_c_min, q_inv_max)
        m_max_bh_bgw, _ = chirp_q_to_comp_masses(m_c_max, q_inv_min)

        # plot!
        smf_gds = gd.MCSamples(samples=smf_samples, \
                               names=['d', 'm_1', 'm_2', 'm_c', 'q_inv'], \
                               labels=[r'D_L', 'm_1', 'm_2', 'M_c', '1/q'], \
                               ranges={'d':(d_min, d_max), \
                                       'm_1':(m_min_bh, m_max_bh), \
                                       'm_2':(m_min_ns, m_max_ns), \
                                       'm_c':(m_c_min, m_c_max), \
                                       'q_inv':(q_inv_min, q_inv_max)}, \
                               label='SMF Priors')
        bgw_gds = gd.MCSamples(samples=bgw_samples, \
                               names=['d', 'm_1', 'm_2', 'm_c', 'q_inv'], \
                               labels=[r'D_L', 'm_1', 'm_2', 'M_c', '1/q'], \
                               ranges={'d':(d_min, d_max), \
                                       'm_1':(m_min_bh_bgw, m_max_bh_bgw), \
                                       'm_2':(m_min_ns_bgw, m_max_ns_bgw), \
                                       'm_c':(m_c_min, m_c_max), \
                                       'q_inv':(q_inv_min, q_inv_max)}, \
                               label='Bilby Priors')
        g = gdp.get_subplot_plotter()
        g.settings.lw_contour = lw
        cm = mpcm.get_cmap('plasma')
        g.triangle_plot([bgw_gds, smf_gds], filled=True, \
                        line_args=[{'lw':lw, 'color':'C0'}, \
                                   {'lw':lw, 'color':'C1'}], \
                        colors=['C0', 'C1'])
        n_pars = smf_samples.shape[1]
        for i in range(n_pars):
            for j in range(0, i + 1):
                g.subplots[i, j].grid(False)
        plot_file = outdir + '/' + label + '_bilby_prior_comp.pdf'
        g.export(plot_file)
        exit()

# calculate tidal deformabilities for all NSs
lambdas = np.zeros(n_inj)
for i in range(n_inj):
    lambdas[i] = dd2_lambda_from_mass(data['mass2'][i])

# determine remnant masses for all mergers using AW's code
remnant_masses = np.zeros(n_inj)
print('calculating remnant masses')
for i in range(n_inj):
    if i % 100 == 0:
        print('{:d}/{:d} masses calculated'.format(i, n_inj))
    sim_id = int(data['simulation_id'][i].split(':')[-1])
    sim = {'mass1': data['mass1'][i], 'spin1x': data['spin1x'][i], \
           'spin1y': data['spin1y'][i], 'spin1z': data['spin1z'][i], \
           'mass2': data['mass2'][i], 'spin2x': data['spin2x'][i], \
           'spin2y': data['spin2y'][i], 'spin2z': data['spin2z'][i]}
    m = nseos.Foucart(sim, eos="DD2")
    remnant_masses[i] = m.remnant_mass()
has_remnant = remnant_masses > 0.0
n_rem = np.sum(has_remnant)

# population plots. start with histograms
fig, axes = mp.subplots(4, 3, figsize=(10, 15))
axes[0, 0].hist(np.diff(data['geocent_end_time'] + \
                        data['geocent_end_time_ns'] * 1.0e-9))
axes[0, 1].hist(data['distance'])
axes[0, 2].hist(data['inclination'])
axes[1, 0].hist(data['latitude'])
axes[1, 1].hist(data['longitude'])
axes[1, 2].hist(data['mass1'])
axes[2, 0].hist(data['mass2'])
axes[2, 1].hist(np.sqrt(data['spin1x'] ** 2 + \
                        data['spin1y'] ** 2 + \
                        data['spin1z'] ** 2))
axes[2, 2].hist(np.sqrt(data['spin2x'] ** 2 + \
                        data['spin2y'] ** 2 + \
                        data['spin2z'] ** 2))
axes[3, 0].hist(data['mass1'] / data['mass2'])
axes[3, 1].hist(remnant_masses[has_remnant])
axes[3, 2].hist(lambdas)

# add labels
axes[0, 0].set_xlabel(r'$\Delta t$')
axes[0, 1].set_xlabel(r'$d$')
axes[0, 2].set_xlabel(r'$\iota$')
axes[1, 0].set_xlabel(r'$\theta$')
axes[1, 1].set_xlabel(r'$\phi$')
axes[1, 2].set_xlabel(r'$m_{\rm BH}$')
axes[2, 0].set_xlabel(r'$m_{\rm NS}$')
axes[2, 1].set_xlabel(r'$|a_{\rm BH}|$')
axes[2, 2].set_xlabel(r'$|a_{\rm NS}|$')
axes[3, 0].set_xlabel(r'$m_{\rm BH} / m_{\rm NS}$')
axes[3, 1].set_xlabel(r'$m_{\rm remnant}>0$')
axes[3, 2].set_xlabel(r'$\Lambda_{\rm NS}$')

# plot expectations
norm = 10.0 / n_inj / 1000.0 # sum to n_inj; 1000-point grids; 10-point hists
axes[0, 0].axvline(s_per_event, color='C1')
axes[0, 0].axvline(np.mean(np.diff(data['geocent_end_time'] + \
                                   data['geocent_end_time_ns'] * 1.0e-9)), \
                   color='C2', ls='--')
d_grid = np.linspace(d_min, 800.0, 1000)
if sample_z:
    z_min = d2z(d_min, h_0, q_0)
    z_max = d2z(d_max, h_0, q_0)
    z_grid = np.linspace(z_min, z_max, 1000)
    dvdd = dvolume_dz(z_grid, h_0, q_0, redshift_rate=redshift_rate) * \
           dz_dd(z_grid, h_0, q_0)
    axes[0, 1].plot(z2d(z_grid, h_0, q_0), dvdd / np.sum(dvdd) / norm)
else:
    dvdd = dvolume_dd(d_grid, h_0, q_0, order=3)
    axes[0, 1].plot(d_grid, dvdd / np.sum(dvdd) / norm)
    dvdd = dvolume_dd(d_grid, h_0, q_0, order=2)
    axes[0, 1].plot(d_grid, dvdd / np.sum(dvdd) / norm, ls='--')
dvdd = dvolume_dd(d_grid, h_0, q_0, order=1)
axes[0, 1].plot(d_grid, dvdd / np.sum(dvdd) / norm, ls='-.')
i_grid = np.linspace(0.0, np.pi, 1000)
axes[0, 2].plot(i_grid, np.sin(i_grid) / np.sum(np.sin(i_grid)) / norm)
theta_grid = np.linspace(-np.pi / 2.0, np.pi / 2.0, 1000)
axes[1, 0].plot(theta_grid, np.cos(theta_grid) / \
                            np.sum(np.cos(theta_grid)) / norm)
axes[1, 1].axvline(0.0, color='C1')
axes[1, 1].axvline(2.0 * np.pi, color='C1')
if uniform_bh_masses:
    axes[1, 2].axvline(m_min_bh, color='C1')
    axes[1, 2].axvline(m_max_bh, color='C1')
else:
    m_grid = np.linspace(m_min_bh, m_mean_bh + 4.0 * m_std_bh, 1000)
    dndm = np.exp(-0.5 * ((m_grid - m_mean_bh) / m_std_bh) ** 2)
    dndm = dndm / np.sum(dndm) / norm
    axes[1, 2].plot(m_grid, dndm)
if uniform_ns_masses:
    axes[2, 0].axvline(m_min_ns, color='C1')
    axes[2, 0].axvline(m_max_ns, color='C1')
else:
    m_grid = np.linspace(m_min_ns, m_mean_ns + 4.0 * m_std_ns, 1000)
    dndm = np.exp(-0.5 * ((m_grid - m_mean_ns) / m_std_ns) ** 2)
    dndm = dndm / np.sum(dndm) / norm
    axes[2, 0].plot(m_grid, dndm)
axes[2, 1].axvline(spin_min_bh, color='C1')
axes[2, 1].axvline(spin_max_bh, color='C1')
axes[2, 2].axvline(spin_min_ns, color='C1')
axes[2, 2].axvline(spin_max_ns, color='C1')

# finish plot
for i in range(4):
    axes[i, 0].set_ylabel(r'${\rm number}$')
    axes[i, 0].set_yticks([])
    axes[i, 1].get_yaxis().set_visible(False)
    axes[i, 2].get_yaxis().set_visible(False)
    for ax in axes[i, :]:
        ax.grid(False)
fig.subplots_adjust(wspace=0.0)
fig.savefig('data/' + label + '_pars_all.pdf', bbox_inches='tight')
mp.close(fig)

# also plot distances vs mass ratios
fig_dq, axes_dq = mp.subplots(2, 2)
axes_dq[0, 0].hist(data['distance'])
axes_dq[0, 0].get_xaxis().set_visible(False)
axes_dq[0, 0].get_yaxis().set_visible(False)
axes_dq[0, 0].set_title(r'$d_L\,{\rm [Mpc]}$')
axes_dq[0, 0].grid(False)
axes_dq[1, 1].hist(data['mass1'] / data['mass2'])
axes_dq[1, 1].get_yaxis().set_visible(False)
axes_dq[1, 1].set_xlabel(r'$m_{BH} / m_{NS}$')
axes_dq[1, 1].set_title(r'$m_{BH} / m_{NS}$')
axes_dq[1, 1].grid(False)
#axes_dq[1, 0].hist2d(data['distance'], data['mass1'] / data['mass2'], \
#                     bins=10)
axes_dq[1, 0].set_xlabel(r'$d_L\,{\rm [Mpc]}$')
axes_dq[1, 0].set_ylabel(r'$m_{BH} / m_{NS}$')
axes_dq[1, 0].plot(data['distance'], data['mass1'] / data['mass2'], \
                   '+', color='C0', alpha=0.2)
axes_dq[1, 0].set_xlim(axes_dq[0, 0].get_xlim())
axes_dq[1, 0].set_ylim(axes_dq[1, 1].get_xlim())
axes_dq[1, 0].grid(False)
fig_dq.delaxes(axes_dq[0, 1])
fig_dq.subplots_adjust(wspace=0.0, hspace=0.0)
fig_dq.savefig('data/' + label + '_d_q_all.pdf', bbox_inches='tight')
mp.close(fig_dq)

# set up logger?
bilby.core.utils.setup_logger(outdir=outdir, label=label, \
                              log_level='WARNING')

# see what we can detect with bilby!
print('calculating SNRs')
rng_states = []
snrs = np.zeros(n_inj)
for j in range(n_inj):

    # report progress
    if j % 100 == 0:
        print('{:d}/{:d} SNRs calculated'.format(j, n_inj))

    # set up next injection. extract mass and orbital parameters
    mass_1 = data['mass1'][j]
    mass_2 = data['mass2'][j]
    spin_1x = data['spin1x'][j]
    spin_1y = data['spin1y'][j]
    spin_1z = data['spin1z'][j]
    spin_2x = data['spin2x'][j]
    spin_2y = data['spin2y'][j]
    spin_2z = data['spin2z'][j]
    iota = data['inclination'][j]
    phase = data['coa_phase'][j]
    distance = data['distance'][j]
    time = float(data['geocent_end_time'][j] + \
                 data['geocent_end_time_ns'][j] * 1.0e-9)
    pol = data['polarization'][j]
    lon = data['longitude'][j]
    lat = data['latitude'][j]
    lambda_1 = 0.0
    lambda_2 = lambdas[j]

    # convert longitude and latitude (in radians) to RA, DEC (in radians)
    t = at.Time(time, format='gps')
    sc = ac.SkyCoord(lon, lat, obstime=t, unit=au.Unit('rad'))
    ra = sc.ra.rad
    dec = sc.dec.rad

    # convert LALInference spin parameters into form desired by 
    # waveform generators
    converted = inverse_transform_precessing_spins(iota, spin_1x, spin_1y, \
                                                   spin_1z, spin_2x, spin_2y, \
                                                   spin_2z, mass_1, mass_2, \
                                                   reference_frequency, phase)
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = converted
    check_conversion = False
    if check_conversion:
        mass_1_kg = mass_1 * 1.98847e30
        mass_2_kg = mass_2 * 1.98847e30
        inputs = np.array([iota, spin_1x, spin_1y, spin_1z, spin_2x, \
                           spin_2y, spin_2z])
        outputs = \
            np.array(bc.bilby_to_lalsimulation_spins(theta_jn, phi_jl, \
                                                     tilt_1, tilt_2, phi_12, \
                                                     a_1, a_2, mass_1_kg, \
                                                     mass_2_kg, \
                                                     reference_frequency, \
                                                     phase))
        print('percentage error in conversions:')
        print((outputs - inputs) / inputs * 100.0)

    # store random state to ensure exact same data used by detection
    # and inference codes
    rng_states.append(np.random.get_state())

    # We are going to inject a binary neutron star waveform.  We first establish a
    # dictionary of parameters that includes all of the different waveform
    # parameters, including masses of the black hole (mass_1) and NS (mass_2),
    # spins of both objects (a, tilt, phi), and deformabilities (lambdas, 
    # with the BH deformability fixed to zero) etc.
    injection_parameters = dict(mass_1=mass_1, mass_2=mass_2, a_1=a_1, \
                                a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, \
                                phi_12=phi_12, phi_jl=phi_jl, \
                                luminosity_distance=distance, \
                                theta_jn=theta_jn, psi=pol, phase=phase, \
                                geocent_time=time, ra=ra, dec=dec, \
                                lambda_1=lambda_1, lambda_2=lambda_2, \
                                time_jitter=0.0)

    # Fixed arguments passed into the source model
    waveform_arguments = dict(waveform_approximant=waveform_approximant, \
                              reference_frequency=reference_frequency, \
                              minimum_frequency=minimum_frequency)

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    # the generator will convert all the parameters
    waveform_generator = \
        bilby.gw.WaveformGenerator(duration=duration, \
                                   sampling_frequency=sampling_frequency, \
                                   frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star, \
                                   parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters, \
                                   waveform_arguments=waveform_arguments)

    # Set up interferometers.  Default is three interferometers:
    # LIGO-Hanford (H1), LIGO-Livingston (L1) and Virgo. These default to 
    # their design sensitivity
    all_bilby_ifo_list = ['CE', 'ET', 'GEO600', 'H1', 'K1', 'L1', 'V1']
    bilby_ifo_list = sorted(list(set(ifo_list) & set(all_bilby_ifo_list)))
    local_ifo_list = sorted(list(np.setdiff1d(ifo_list, all_bilby_ifo_list, \
                                              assume_unique=True)))
    ifos = bilby.gw.detector.InterferometerList(bilby_ifo_list)
    for ifo in local_ifo_list:
        local_ifo_file = './data/detectors/' + ifo + '.interferometer'
        try:
            ifos.append(bd.load_interferometer(local_ifo_file))
        except OSError:
            raise ValueError('Interferometer ' + ifo + ' not implemented')
    ifos._check_interferometers()
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration,
        start_time=injection_parameters['geocent_time'] + 2 - duration)
    ifos.inject_signal(waveform_generator=waveform_generator, \
                       parameters=injection_parameters)

    # extract SNRs for each detector and form a network SNR
    opt_snrs = []
    mf_snrs = []
    for ifo in ifos:
        opt_snrs.append(ifo.meta_data['optimal_SNR'])
        mf_snrs.append(abs(ifo.meta_data['matched_filter_SNR']))
    opt_snrs = np.array(opt_snrs)
    mf_snrs = np.array(mf_snrs)
    snrs[j] = np.sqrt(np.sum(mf_snrs ** 2))

# define detected sample
det = snrs > snr_thresh
n_det = np.sum(det)
det_rem = np.logical_and(has_remnant, det)
n_det_rem = np.sum(det_rem)

# save clean data file and plot SNRs vs distance and inclination
fig_snr, axes_snr = mp.subplots(1, 2, figsize=(10, 5), sharey=True)
axes_snr[0].semilogy(data['distance'], snrs, '.')
axes_snr[0].set_xlabel(r'$d$')
axes_snr[0].set_ylabel(r'$\rho$')
axes_snr[0].grid(False)
axes_snr[0].axhline(snr_thresh, color='C1', ls='--')
axes_snr[1].semilogy(data['inclination'], snrs, '.')
axes_snr[1].set_xlabel(r'$\iota$')
axes_snr[1].grid(False)
axes_snr[1].axhline(snr_thresh, color='C1', ls='--')
fig_snr.subplots_adjust(wspace=0.0)
fig_snr.savefig(os.path.join(outdir, label + '_snrs.pdf'), bbox_inches='tight')
mp.close(fig_snr)

# plot parameter distributions for detected events
fig, axes = mp.subplots(4, 3, figsize=(10, 15))
axes[0, 0].hist(np.diff(data['geocent_end_time'][det] + \
                        data['geocent_end_time_ns'][det] * 1.0e-9))
axes[0, 1].hist(data['distance'][det])
axes[0, 2].hist(data['inclination'][det])
axes[1, 0].hist(data['latitude'][det])
axes[1, 1].hist(data['longitude'][det])
axes[1, 2].hist(data['mass1'][det])
axes[2, 0].hist(data['mass2'][det])
axes[2, 1].hist(np.sqrt(data['spin1x'][det] ** 2 + \
                        data['spin1y'][det] ** 2 + \
                        data['spin1z'][det] ** 2))
axes[2, 2].hist(np.sqrt(data['spin2x'][det] ** 2 + \
                        data['spin2y'][det] ** 2 + \
                        data['spin2z'][det] ** 2))
axes[3, 0].hist(data['mass1'][det] / data['mass2'][det])
axes[3, 1].hist(remnant_masses[det_rem])
axes[3, 2].hist(lambdas[det])
axes[0, 0].set_xlabel(r'$\Delta t$')
axes[0, 1].set_xlabel(r'$d$')
axes[0, 2].set_xlabel(r'$\iota$')
axes[1, 0].set_xlabel(r'$\theta$')
axes[1, 1].set_xlabel(r'$\phi$')
axes[1, 2].set_xlabel(r'$m_{\rm BH}$')
axes[2, 0].set_xlabel(r'$m_{\rm NS}$')
axes[2, 1].set_xlabel(r'$|a_{\rm BH}|$')
axes[2, 2].set_xlabel(r'$|a_{\rm NS}|$')
axes[3, 0].set_xlabel(r'$m_{\rm BH} / m_{\rm NS}$')
axes[3, 1].set_xlabel(r'$m_{\rm remnant}>0$')
axes[3, 2].set_xlabel(r'$\Lambda_{\rm NS}$')
norm = 10.0 / n_det / 1000.0 # sum to n_inj; 1000-point grids; 10-point hists
axes[0, 0].axvline(s_per_event * n_inj / n_det, color='C1')
axes[0, 0].axvline(np.mean(np.diff(data['geocent_end_time'][det] + \
                                   data['geocent_end_time_ns'][det] * 1.0e-9)), \
                   color='C2', ls='--')
d_grid = np.linspace(0.0, 800.0, 1000)
if sample_z:
    z_min = d2z(d_min, h_0, q_0)
    z_max = d2z(d_max, h_0, q_0)
    z_grid = np.linspace(z_min, z_max, 1000)
    dvdd = dvolume_dz(z_grid, h_0, q_0, redshift_rate=redshift_rate) * \
           dz_dd(z_grid, h_0, q_0)
    axes[0, 1].plot(z2d(z_grid, h_0, q_0), dvdd / np.sum(dvdd) / norm)
else:
    dvdd = dvolume_dd(d_grid, h_0, q_0, order=3)
    axes[0, 1].plot(d_grid, dvdd / np.sum(dvdd) / norm)
i_grid = np.linspace(0.0, np.pi, 1000)
#norm = i_grid[1] - i_grid[0]
axes[0, 2].plot(i_grid, np.sin(i_grid) / np.sum(np.sin(i_grid)) / norm)
theta_grid = np.linspace(-np.pi / 2.0, np.pi / 2.0, 1000)
#norm = theta_grid[1] - theta_grid[0]
axes[1, 0].plot(theta_grid, np.cos(theta_grid) / \
                            np.sum(np.cos(theta_grid)) / norm)
axes[1, 1].axvline(0.0, color='C1')
axes[1, 1].axvline(2.0 * np.pi, color='C1')
if uniform_bh_masses:
    axes[1, 2].axvline(m_min_bh, color='C1')
    axes[1, 2].axvline(m_max_bh, color='C1')
else:
    m_grid = np.linspace(m_min_bh, m_mean_bh + 4.0 * m_std_bh, 1000)
    dndm = np.exp(-0.5 * ((m_grid - m_mean_bh) / m_std_bh) ** 2)
    dndm = dndm / np.sum(dndm) / norm
    axes[1, 2].plot(m_grid, dndm)
if uniform_ns_masses:
    axes[2, 0].axvline(m_min_ns, color='C1')
    axes[2, 0].axvline(m_max_ns, color='C1')
else:
    m_grid = np.linspace(m_min_ns, m_mean_ns + 4.0 * m_std_ns, 1000)
    #norm = m_grid[1] - m_grid[0]
    dndm = np.exp(-0.5 * ((m_grid - m_mean_ns) / m_std_ns) ** 2)
    dndm = dndm / np.sum(dndm) / norm
    axes[2, 0].plot(m_grid, dndm)
axes[2, 1].axvline(spin_min_bh, color='C1')
axes[2, 1].axvline(spin_max_bh, color='C1')
axes[2, 2].axvline(spin_min_ns, color='C1')
axes[2, 2].axvline(spin_max_ns, color='C1')
for i in range(4):
    axes[i, 0].set_ylabel(r'${\rm number}$')
    axes[i, 0].set_yticks([])
    axes[i, 1].get_yaxis().set_visible(False)
    axes[i, 2].get_yaxis().set_visible(False)
    for ax in axes[i, :]:
        ax.grid(False)
fig.subplots_adjust(wspace=0.0)
fig.savefig('data/' + label + '_pars_det.pdf', bbox_inches='tight')
mp.close(fig)

# also plot distances vs mass ratios for detected events
fig_dq, axes_dq = mp.subplots(2, 2)
axes_dq[0, 0].hist(data['distance'][det], \
                   density=True)
axes_dq[0, 0].hist(data['distance'][det_rem], \
                   density=True, color='C1', alpha=0.7)
axes_dq[0, 0].get_xaxis().set_visible(False)
axes_dq[0, 0].get_yaxis().set_visible(False)
axes_dq[0, 0].set_title(r'$d_L\,{\rm [Mpc]}$')
axes_dq[0, 0].grid(False)
axes_dq[1, 1].hist(data['mass1'][det] / data['mass2'][det], \
                   density=True, label='no remnant')
axes_dq[1, 1].hist(data['mass1'][det_rem] / data['mass2'][det_rem], \
                   density=True, color='C1', alpha=0.7, label='remnant')
axes_dq[1, 1].get_yaxis().set_visible(False)
axes_dq[1, 1].set_xlabel(r'$m_{BH} / m_{NS}$')
axes_dq[1, 1].set_title(r'$m_{BH} / m_{NS}$')
axes_dq[1, 1].grid(False)
axes_dq[1, 1].legend(loc='upper right')
#axes_dq[1, 0].hist2d(data['distance'], data['mass1'] / data['mass2'], \
#                     bins=10)
axes_dq[1, 0].set_xlabel(r'$d_L\,{\rm [Mpc]}$')
axes_dq[1, 0].set_ylabel(r'$m_{BH} / m_{NS}$')
axes_dq[1, 0].plot(data['distance'][det], \
                   data['mass1'][det] / data['mass2'][det], \
                   '+', color='C0', alpha=0.2)
axes_dq[1, 0].plot(data['distance'][det_rem], \
                   data['mass1'][det_rem] / data['mass2'][det_rem], \
                   '+', color='C1', alpha=0.7)
axes_dq[1, 0].set_xlim(axes_dq[0, 0].get_xlim())
axes_dq[1, 0].set_ylim(axes_dq[1, 1].get_xlim())
axes_dq[1, 0].grid(False)
fig_dq.delaxes(axes_dq[0, 1])
fig_dq.subplots_adjust(wspace=0.0, hspace=0.0)
fig_dq.savefig('data/' + label + '_d_q_det.pdf', bbox_inches='tight')
mp.close(fig_dq)

# save everything to file
fmt = '{:s},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},' + \
      '{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:d},' + \
      '{:d},{:.9e},{:.9e},{:.9e}'
with open('data/' + label + '.txt', 'w') as f:
    f.write('#' + ','.join(to_store) + ',lambda_2,remnant_mass,snr')
    for i in range(n_inj):
        f.write('\n' + \
                fmt.format(data['simulation_id'][i], data['mass1'][i], \
                           data['mass2'][i], data['spin1x'][i], \
                           data['spin1y'][i], data['spin1z'][i], \
                           data['spin2x'][i], data['spin2y'][i], \
                           data['spin2z'][i], data['redshift'][i], \
                           data['distance'][i], data['inclination'][i], \
                           data['coa_phase'][i], data['polarization'][i], \
                           data['longitude'][i], data['latitude'][i], \
                           data['geocent_end_time'][i], \
                           data['geocent_end_time_ns'][i], lambdas[i], \
                           remnant_masses[i], snrs[i]))

# save random states to file
with open('data/' + label + '_rng_states.bin', 'wb') as f:
    pickle.dump(rng_states, f)

# final report
print(n_inj, 'mergers simulated')
print(n_det, 'mergers detected')
print(n_rem, 'mergers with non-zero remnant mass')
print(n_det_rem, 'mergers detected with non-zero remnant mass')
