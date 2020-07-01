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
import sys

def allocate_jobs(n_jobs, n_procs=1, rank=0):
    n_j_allocated = 0
    for i in range(n_procs):
        n_j_remain = n_jobs - n_j_allocated
        n_p_remain = n_procs - i
        n_j_to_allocate = n_j_remain // n_p_remain
        if rank == i:
            return range(n_j_allocated, \
                         n_j_allocated + n_j_to_allocate)
        n_j_allocated += n_j_to_allocate

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
                    aligned_spins=False, z_min=None, z_max=None):

    # constrained realisation if desired
    if seed is not None:
        npr.seed(seed)

    # first draw number of events: a Poisson process, with rate 
    # given by the number of events per year per Gpc^3, the 
    # duration of observations and the volume
    if fixed_count is None:
        if sample_z:
            if z_min is None:
                z_min = d2z(d_min, h_0, q_0)
            if z_max is None:
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
            if z_min is None:
                z_min = d2z(d_min, h_0, q_0)
            if z_max is None:
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
              ('spin2y', float), ('spin2z', float), \
              ('distance', float), ('inclination', float), \
              ('coa_phase', float), ('polarization', float), \
              ('longitude', float), ('latitude', float), \
              ('geocent_end_time', int), ('geocent_end_time_ns', int)]
    if sample_z:
        dtypes.append(('redshift', float))
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
    data['distance'] = distances
    data['inclination'] = incs
    data['coa_phase'] = coa_phases
    data['polarization'] = pols
    data['longitude'] = longs
    data['latitude'] = colats
    data['geocent_end_time'] = [int(math.modf(t)[1]) for t in times]
    data['geocent_end_time_ns'] = [int(math.modf(t)[0] * 1e9) for t in times]
    if sample_z:
        data['redshift'] = redshifts

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
h_0_fid = 67.36 # km / s / Mpc
q_0_fid = 0.5 * 0.3153 - 0.6847
rate = 6.1e-10 # 6.1e-7 # 2.0e-6 # events / year / Mpc^3
d_min = 0.0
t_start = 1325030418
t_stop = t_start + t_obs * 3600 * 24 * 365
f_online = 0.5
seed_1 = 141023
seed_2 = 161222
to_store = ['simulation_id', 'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', \
            'spin2x' , 'spin2y', 'spin2z', 'distance', 'inclination', \
            'coa_phase', 'polarization', 'longitude', 'latitude', \
            'geocent_end_time', 'geocent_end_time_ns']
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

# set up identical within-chain MPI processes
use_mpi = False
if use_mpi:
    import mpi4py.MPI as mpi
    n_procs = mpi.COMM_WORLD.Get_size()
    rank = mpi.COMM_WORLD.Get_rank()
elif len(sys.argv) > 1:
    if len(sys.argv) == 3:
        n_procs = int(sys.argv[1])
        rank = int(sys.argv[2])
        if rank > n_procs or rank < 1:
            exit('ERROR: 1 <= rank <= number of processes.')
        rank -= 1    
    else:
        exit('ERROR: please call using ' + \
             '"python n_bar_det_farming.py <n_procs> <rank>" format ' + \
             'to specify number of processes and rank without MPI. ' + \
             'NB: rank should be one-indexed.')
else:
    n_procs = 1
    rank = 0

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

# convert distance limits to redshift limits
z_min_fid = d2z(d_min, h_0_fid, q_0_fid)
z_max_fid = d2z(d_max, h_0_fid, q_0_fid)

# set out grid of parameters
n_grid = 5
n_rpts = 3
n_jobs = n_grid * n_grid * n_rpts
h_0_min, h_0_max = 60.0, 80.0
q_0_min, q_0_max = -2.0, 1.0
h_0_grid = np.linspace(h_0_min, h_0_max, n_grid)
q_0_grid = np.linspace(q_0_min, q_0_max, n_grid)

# assign jobs
job_list = allocate_jobs(n_jobs, n_procs, rank)

# set up structured array for outputs
dtypes = [('i_job', int), ('i_h_0', int), ('i_q_0', int), \
          ('h_0', float), ('q_0', float), ('n_inj', int), \
          ('n_det', int), ('n_det_rem', int)]
if sample_z:
    dtypes.append(('z_max_det', float))
    dtypes.append(('z_max_det_rem', float))
else:
    dtypes.append(('d_max_det', float))
    dtypes.append(('d_max_det_rem', float))
stats = np.empty((len(job_list), ), dtype=dtypes)

# loop over jobs
print('process {:d} jobs: '.format(rank), job_list)
for i_job in job_list:

    # determine grid indices. jobs are ordered such that first 
    # n_grid ** 2 jobs loop through the flattened parameter grid,
    # the next n_grid ** 2 jobs are repeats with different seeds,
    # etc.
    print('processing job {:d}'.format(i_job))
    i_rpt = i_job // n_grid ** 2
    i_h_0 = (i_job - i_rpt * n_grid ** 2) // n_grid
    i_q_0 = (i_job - i_rpt * n_grid ** 2) - i_h_0 * n_grid

    # generate a random seed for each realisation. use the same 
    # seed for all cosmologies within a repeat to reduce sample 
    # variance
    seed = seed_1 + i_rpt * seed_2

    # simulate using my code
    h_0 = h_0_grid[i_h_0]
    q_0 = q_0_grid[i_q_0]
    pop = nsbh_population(rate, t_start, t_stop, f_online, d_min, \
                          d_max, h_0, q_0, m_min_bh, m_max_bh, \
                          m_mean_bh, m_std_bh, m_min_ns, m_max_ns, \
                          m_mean_ns, m_std_ns, spin_min_bh, \
                          spin_max_bh, spin_min_ns, spin_max_ns, \
                          seed=seed, sample_z=sample_z, \
                          redshift_rate=redshift_rate, \
                          uniform_bh_masses=uniform_bh_masses, \
                          uniform_ns_masses=uniform_ns_masses, \
                          aligned_spins=aligned_spins, \
                          z_min=z_min_fid, z_max=z_max_fid)
    s_per_event = pop[0]
    data = pop[1]
    n_inj = data.shape[0]

    # calculate tidal deformabilities for all NSs
    lambdas = np.zeros(n_inj)
    for i in range(n_inj):
        lambdas[i] = dd2_lambda_from_mass(data['mass2'][i])

    # see what we can detect with bilby!
    #print('calculating SNRs')
    snrs = np.zeros(n_inj)
    remnant_masses = np.zeros(n_inj)
    for j in range(n_inj):

        ## report progress
        #if j % 100 == 0:
        #    print('{:d}/{:d} SNRs calculated'.format(j, n_inj))

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
        bilby.core.utils.setup_logger(log_level='WARNING')
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

        # if detected, calculate remnant mass
        if snrs[j] > snr_thresh:

            sim = {'mass1': data['mass1'][j], 'spin1x': data['spin1x'][j], \
                   'spin1y': data['spin1y'][j], 'spin1z': data['spin1z'][j], \
                   'mass2': data['mass2'][j], 'spin2x': data['spin2x'][j], \
                   'spin2y': data['spin2y'][j], 'spin2z': data['spin2z'][j]}
            m = nseos.Foucart(sim, eos="DD2")
            remnant_masses[j] = m.remnant_mass()
        
        else:

            remnant_masses[j] = -1.0

    # define detected sample
    det = snrs > snr_thresh
    n_det = np.sum(det)
    has_remnant = remnant_masses > 0.0
    n_rem = np.sum(has_remnant)
    det_rem = np.logical_and(has_remnant, det)
    n_det_rem = np.sum(det_rem)

    # store stats
    i_store = i_job - i_rpt * n_grid ** 2
    stats['i_job'][i_store] = i_job
    stats['i_h_0'][i_store] = i_h_0
    stats['i_q_0'][i_store] = i_q_0
    stats['h_0'][i_store] = h_0
    stats['q_0'][i_store] = q_0
    stats['n_inj'][i_store] = n_inj
    stats['n_det'][i_store] = n_det
    stats['n_det_rem'][i_store] = n_det_rem
    if sample_z:
        if n_det == 0:
            stats['z_max_det'][i_store] = 0.0
        else:
            stats['z_max_det'][i_store] = np.max(data['redshift'][det])
        if n_det_rem == 0:
            stats['z_max_det_rem'][i_store] = 0.0
        else:
            stats['z_max_det_rem'][i_store] = np.max(data['redshift'][det_rem])
    else:
        if n_det == 0:
            stats['d_max_det'][i_store] = 0.0
        else:
            stats['d_max_det'][i_store] = np.max(data['distance'][det])
        if n_det_rem == 0:
            stats['d_max_det_rem'][i_store] = 0.0
        else:
            stats['d_max_det_rem'][i_store] = np.max(data['distance'][det_rem])

# save to file
fname = 'data/' + label + \
        '_n_det_proc_{:d}_of_{:d}.txt'.format(rank, n_procs)
np.savetxt(fname, stats, delimiter=',', \
           header=','.join(stats.dtype.names), \
           fmt='%d,%d,%d,%f,%f,%d,%d,%d,%f,%f')

