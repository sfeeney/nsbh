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
import lalsimulation as lalsim
import astropy.time as at
import astropy.coordinates as ac
import astropy.units as au

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


# @TODO
# 1 - DONE: set up MPI
# 2 - DONE: read in parameters
# 3 - DONE: calculate deformabilities
# 4 - DONE: convert AW's parameters to bilby for injection
# 5 - DONE: set up prior: BNS w/ BBH mass range for m1, lambda1 fixed to zero
# 6 - check all settings
# 7 - check out psi definition. prior is 0->pi but injection is 4.03...
# 8 - fix angular positions?

# settings
use_mpi = True
duration = 8.0 # 32.0
sampling_frequency = 2048.
minimum_frequency = 40.0 # 20.0
reference_frequency = 14.0 # 50.0
zero_spins = False
remnants_only = True

# set up identical within-chain MPI processes
if use_mpi:
    import mpi4py.MPI as mpi
    n_procs = mpi.COMM_WORLD.Get_size()
    rank = mpi.COMM_WORLD.Get_rank()
else:
    n_procs = 1
    rank = 0

# read list of all targets and assign
targets = np.genfromtxt('data/remnant_sorted_detected.txt', delimiter=' ')
target_ids = targets[:, 0].astype(int)
if remnants_only:
    target_ids = target_ids[targets[:, 1] > 0.0]
n_inj = len(target_ids)
job_list = allocate_jobs(n_inj, n_procs, rank)

# loop over assignments
for job in job_list:

    # which injection?
    # read in injection parameters from Andrew
    # entries are simulation_id, mass1, mass2, spin1x, spin1y, spin1z, 
    # spin2x, spin2y, spin2z, distance, inclination, coa_phase, 
    # polarization, longitude, latitude, geocent_end_time, geocent_end_time_ns
    # can access these names using raw_pars.dtype.names
    raw_pars = np.genfromtxt('data/NSBH_samples_precessing_DD2_detected.dat', \
                             dtype=None, names=True, delimiter=',', \
                             encoding=None)
    search_str = 'sim_inspiral:simulation_id:{:d}'.format(target_ids[job])
    inj_id = np.argwhere(raw_pars['simulation_id']==search_str)[0, 0]
    sel_pars = raw_pars[inj_id]

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
    time = float(sel_pars['geocent_end_time'])
    pol = sel_pars['polarization']
    lon = sel_pars['longitude']
    lat = sel_pars['latitude']
    lambda_1 = 0.0
    lambda_2 = dd2_lambda_from_mass(mass_2)

    # optionally zero spin parameters
    if zero_spins:
        spin_1x = 0.0
        spin_1y = 0.0
        spin_1z = 0.0
        spin_2x = 0.0
        spin_2y = 0.0
        spin_2z = 0.0

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
    label_str = 'nsbh_inj_{:d}_d_{:04.1f}_mf_{:4.1f}_rf_{:4.1f}'
    label = label_str.format(inj_id, duration, minimum_frequency, \
                             reference_frequency)
    if zero_spins:
        label += '_zero_spins'
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    # Set up a random seed for result reproducibility.  This is optional!
    np.random.seed(141023)

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

    # Set up interferometers.  In this case we'll use three interferometers:
    # LIGO-Hanford (H1), LIGO-Livingston (L1) and Virgo. These default to 
    # their design sensitivity
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
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
    to_fix = ['lambda_1']
    if zero_spins:
        to_fix += ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl']
    else:
        priors.pop('a_1')
        priors['a_1'] = \
            bilby.core.prior.Uniform(name='a_1', minimum=0, maximum=0.8, \
                                     boundary='reflective')
    for key in to_fix:
        priors[key] = injection_parameters[key]

    # Initialise the likelihood by passing in the interferometer data (ifos) and
    # the waveoform generator, as well the priors.
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
                               sampler='dynesty', npoints=1000, \
                               injection_parameters=injection_parameters, \
                               outdir=outdir, label=label, \
                               conversion_function=bilby.gw.conversion.generate_all_bns_parameters)

    # Make a corner plot.
    result.plot_corner()
