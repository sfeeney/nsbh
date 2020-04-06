#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a full 15 parameter
space for an injected cbc signal. This is the standard injection analysis script
one can modify for the study of injected CBC events.
"""
from __future__ import division, print_function
import numpy as np
import bilby
import bilby.gw.utils as bu
import bilby.gw.conversion as bc
import bilby.gw.detector as bd
import lalsimulation as lalsim
import astropy.time as at
import astropy.coordinates as ac
import astropy.units as au
import sys
import pickle

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

def allocate_all_jobs(n_jobs, n_procs=1):
    allocation = []
    n_j_allocated = 0
    for i in range(n_procs):
        n_j_remain = n_jobs - n_j_allocated
        n_p_remain = n_procs - i
        n_j_to_allocate = n_j_remain // n_p_remain
        allocation.append(range(n_j_allocated, \
                                n_j_allocated + n_j_to_allocate))
        n_j_allocated += n_j_to_allocate
    return allocation

# LAL detector azimuth angles measured clockwise from north. bilby 
# azimuth angles measured anticlockwise from east. helpful link for 
# checking: https://www.ligo.org/scientists/GW100916/detectors.txt
def convert_azimuth(azi):

    new = np.pi / 2.0 - azi
    if new < 0.0:
        new += 2.0 * np.pi

    return new


# @TODO
# 7 - check out psi definition. prior is 0->pi but injection is 4.03...
# 10 - modify chirp mass and 1/q priors to reflect new mass BH prior?
# 14 - check bilby.gw.prior.UniformSourceFrame. redshifted rate? h_0/q_0?
#      likely it's not going to be consistent

# NB: broadening mass prior reduces mergers with non-zero remnant

# settings
use_mpi = False
snr_thresh = 12.0
duration = 32.0 # 8.0
sampling_frequency = 2048.
minimum_frequency = 20.0 # 40.0
reference_frequency = 14.0 # 50.0
ifo_list = ['H1', 'L1', 'V1', 'K1', 'IndIGO'] # ['H1', 'L1', 'V1']
n_live = 1000
zero_spins = False
remnants_only = True
tight_loc = True
fixed_ang = True
sample_z = True
redshift_rate = True
broad_bh_masses = True

# settings for tight localisation: known angular position, distance 
# constrained to be approximately Gaussian by redshift, peculiar 
# velocity and Hubble Constant measurements
c = 2.998e5
sig_z_obs = 0.001
sig_v_pec_obs = 200.0
sig_v_rec_obs = np.sqrt((c * sig_z_obs) ** 2 + sig_v_pec_obs ** 2)
h_0_obs = 67.4
sig_h_0_obs = 0.50

# set up identical within-chain MPI processes
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
             '"python sim_nsbh_analysis.py <n_procs> <rank>" format ' + \
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
if broad_bh_masses:
    label_str += '_bbhmp'
base_label = label_str.format(duration, minimum_frequency, \
                              reference_frequency)

# read in injections from file, select which to process, and assign
# to processes
par_file = base_label + '.txt'
raw_pars = np.genfromtxt('data/' + par_file, \
                         dtype=None, names=True, delimiter=',', \
                         encoding=None)
det = raw_pars['snr'] >= snr_thresh
if remnants_only:
    det = np.logical_and(det, raw_pars['remnant_mass'] > 0.0)
raw_pars = raw_pars[det]
n_inj = np.sum(det)
job_list = allocate_jobs(n_inj, n_procs, rank)

# read in random states used to generate data when calculating 
# SNRs. this way, detection and inference data are the same.
with open('data/' + base_label + '_rng_states.bin', 'rb') as f:
    all_rng_states = pickle.load(f)
rng_states = []
for i in range(len(det)):
    if det[i]:
        rng_states.append(all_rng_states[i])

# generate random deviates used to generate noisy distance estimates
# if required
if tight_loc:
    np.random.seed(141023)
    frac_delta_d_obs = np.random.randn(n_inj)

# loop over assignments
for j in range(len(job_list)):

    # find next injection
    # entries are simulation_id, mass1, mass2, spin1x, spin1y, spin1z, 
    # spin2x, spin2y, spin2z, distance, inclination, coa_phase, 
    # polarization, longitude, latitude, geocent_end_time, 
    # geocent_end_time_ns, remnant_mass and snr
    # can access these names using raw_pars.dtype.names
    sel_pars = raw_pars[job_list[j]]
    inj_id = int(sel_pars['simulation_id'].split(':')[-1])

    # extract mass and orbital parameters
    mass_1 = sel_pars['mass1']
    mass_2 = sel_pars['mass2']
    spin_1x = sel_pars['spin1x']
    spin_1y = sel_pars['spin1y']
    spin_1z = sel_pars['spin1z']
    spin_2x = sel_pars['spin2x']
    spin_2y = sel_pars['spin2y']
    spin_2z = sel_pars['spin2z']
    iota = sel_pars['inclination']
    phase = sel_pars['coa_phase']
    distance = sel_pars['distance']
    time = float(sel_pars['geocent_end_time']) + \
           sel_pars['geocent_end_time_ns'] * 1.0e-9
    pol = sel_pars['polarization']
    lon = sel_pars['longitude']
    lat = sel_pars['latitude']
    lambda_1 = 0.0
    lambda_2 = sel_pars['lambda_2']

    # optionally zero spin parameters
    if zero_spins:
        spin_1x = 0.0
        spin_1y = 0.0
        spin_1z = 0.0
        spin_2x = 0.0
        spin_2y = 0.0
        spin_2z = 0.0

    # if constraining distance from cosmology and redshift, determine
    # the distance uncertainty and draw a noisy distance estimate. this 
    # estimate, along with the distance uncertainty, defines the prior 
    # on distance we will use
    if tight_loc:
        sig_d_obs = np.sqrt((sig_h_0_obs / h_0_obs) ** 2 + \
                            (sig_v_rec_obs / h_0_obs / distance) ** 2) * \
                    distance
        d_obs = distance + frac_delta_d_obs[job_list[j]] * sig_d_obs

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
    check_conversion = True
    if check_conversion and not zero_spins:
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

    # Specify the output directory and the name of the simulation.
    outdir = 'outdir'
    label = base_label + '_inj_{:d}'.format(inj_id)
    if zero_spins:
        label += '_zero_spins'
    if tight_loc:
        label += '_tight_loc'
    elif fixed_ang:
        label += '_fixed_ang'
    if n_live != 1000:
        label += '_nlive_{:04d}'.format(n_live)
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    # set random state to ensure exact same data used by detection
    # and inference codes
    np.random.set_state(rng_states[job_list[j]])

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
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2_NRTidal', \
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
        local_ifo_file = './data/' + ifo + '.interferometer'
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
    
    # Load the default prior for binary neutron stars.
    # We're going to sample in chirp_mass, mass_ratio and lambda_2 for now.
    # BNS have aligned spins by default, so allow precessing spins
    # pass aligned_spin=False to the BNSPriorDict
    priors = bilby.gw.prior.BNSPriorDict(aligned_spin=False)
    priors.pop('mass_1')
    priors.pop('mass_2')
    priors['chirp_mass'] = bilby.prior.Uniform(name='chirp_mass', \
                                               unit='$M_{\\odot}$', \
                                               latex_label='$M$', \
                                               minimum=1.8, maximum=7.2)
    priors['mass_ratio'] = bilby.prior.Uniform(name='mass_ratio', \
                                               latex_label='$q$', \
                                               minimum=0.02, maximum=0.4)
    priors['geocent_time'] = \
        bilby.core.prior.Uniform(minimum=injection_parameters['geocent_time'] - 0.1, \
                                 maximum=injection_parameters['geocent_time'] + 0.1, \
                                 name='geocent_time', latex_label='$t_c$', \
                                 unit='$s$')
    priors['lambda_2'] = bilby.core.prior.Uniform(name='lambda_2', \
                                                  minimum=0.0, \
                                                  maximum=4000.0, \
                                                  latex_label=r'$\Lambda_2$', \
                                                  boundary=None)
    to_fix = ['lambda_1']
    if zero_spins:
        to_fix += ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl']
    else:
        priors.pop('a_1')
        priors['a_1'] = \
            bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=0.8, \
                                     boundary='reflective')
    if tight_loc:
        to_fix += ['ra', 'dec']
        priors.pop('luminosity_distance')
        priors['luminosity_distance'] = \
            bilby.prior.TruncatedGaussian(d_obs, sig_d_obs, 10.0, 5000.0, \
                                          name='luminosity_distance', \
                                          unit='Mpc', latex_label='$D_L$')
    elif fixed_ang:
        to_fix += ['ra', 'dec']
        priors.pop('luminosity_distance')
        priors['luminosity_distance'] = \
            bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', \
                                              minimum=10.0, maximum=5000.0, \
                                              unit='Mpc', boundary=None)
    for key in to_fix:
        priors[key] = injection_parameters[key]

    # Initialise the likelihood by passing in the interferometer data (ifos) and
    # the waveform generator, as well as the priors.
    # The explicit time, distance, and phase marginalizations are turned on to
    # improve convergence, and the parameters are recovered by the conversion
    # function.
    likelihood = \
        bilby.gw.GravitationalWaveTransient(interferometers=ifos, \
            waveform_generator=waveform_generator, priors=priors, \
            distance_marginalization=True, phase_marginalization=True, \
            time_marginalization=True, jitter_time=True)

    # Run sampler. In this case we're going to use the `cpnest` sampler
    # Note that the maxmcmc parameter is increased so that between each iteration of
    # the nested sampler approach, the walkers will move further using an mcmc
    # approach, searching the full parameter space.
    # The conversion function will determine the distance, phase and coalescence
    # time posteriors in post processing.
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, \
                               sampler='dynesty', npoints=n_live, \
                               injection_parameters=injection_parameters, \
                               outdir=outdir, label=label, \
                               conversion_function=bilby.gw.conversion.generate_all_bns_parameters)

    # Make a corner plot.
    result.plot_corner()
