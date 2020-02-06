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

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 8.0 # 32.0
sampling_frequency = 2048.
minimum_frequency = 40.0 # 20.0
reference_frequency = 50.0
sampler = 'dynesty' # 'emcee'
zero_spins = False
minimal_run = False
pre_marge = True
no_td = False

# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label_str = 'nsbh_inj_bilby_d_{:04.1f}_mf_{:4.1f}_rf_{:4.1f}_{:s}'
label = label_str.format(duration, minimum_frequency, reference_frequency, \
                         sampler)
if zero_spins:
    label += '_zero_spin'
if minimal_run:
    label += '_mini'
else:
    if pre_marge:
        label += '_dpt_marge'
if no_td:
    label += '_no_td'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
np.random.seed(22122016)

# We are going to inject a binary neutron star waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the black hole (mass_1) and NS (mass_2),
# spins of both objects (a, tilt, phi), and deformabilities (lambdas, 
# with the BH deformability fixed to zero) etc.
# mass_1=1.5, mass_2=1.3, luminosity_distance=50.
# chi_1=0.02, chi_2=0.02
injection_parameters = dict(mass_1=7.39, mass_2=1.37, \
                            luminosity_distance=105.2307, \
                            theta_jn=0.4, psi=2.659, phase=1.3, \
                            geocent_time=1126259642.413, ra=1.375, \
                            dec=-1.2108, lambda_1=0, lambda_2=450)
if not no_td:
    injection_parameters['lambda_1'] = 0.0
    injection_parameters['lambda_2'] = 450.0
if zero_spins:
    injection_parameters['chi_1'] = 0.0
    injection_parameters['chi_2'] = 0.0
else:
    injection_parameters['chi_1'] = 0.1
    injection_parameters['chi_2'] = 0.02
if not minimal_run and pre_marge:
    injection_parameters['time_jitter'] = 0.0

# @TODO: what is PSI? assumed polarization
# WHAT DURATION: think i've used too much time, cut it down
# SAMPLE FULL PARAM RANGE
# SAMPLE FULL RANGE IN CHI (dropped amp from 0.8 to 0.3)

# different waveform generators based on tidal deformability choices
if no_td:
    waveform_approximant='IMRPhenomPv2'
    fd_source_model=bilby.gw.source.lal_binary_black_hole
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
else:
    waveform_approximant='IMRPhenomPv2_NRTidal'
    fd_source_model=bilby.gw.source.lal_binary_neutron_star
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
waveform_arguments = dict(waveform_approximant=waveform_approximant, \
                          reference_frequency=reference_frequency, \
                          minimum_frequency=minimum_frequency)
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=fd_source_model,
    parameter_conversion=parameter_conversion,
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
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)

# Load the default prior for binary neutron stars.
# We're going to sample in chirp_mass, mass_ratio and lambda_2 for now.
# BNS have aligned spins by default, so allow precessing spins
# pass aligned_spin=False to the BNSPriorDict
priors = bilby.gw.prior.BNSPriorDict(aligned_spin=True)
priors.pop('mass_1')
priors.pop('mass_2')
priors['chirp_mass'] = bilby.prior.Uniform(name='chirp_mass', \
                                           unit='$M_{\\odot}$', \
                                           latex_label='$M$', \
                                           minimum=1.8, maximum=7.2)
priors['mass_ratio'] = bilby.prior.Uniform(name='mass_ratio', \
                                           latex_label='$q$', \
                                           minimum=0.02, maximum=0.4)
if no_td:
    priors.pop('lambda_1')
    priors.pop('lambda_2')
    to_fix = []
else:
    to_fix = ['lambda_1']
if minimal_run:
    to_fix += ['psi', 'geocent_time', 'ra', 'dec', 'theta_jn', \
              'luminosity_distance', 'phase']
else:
    priors['geocent_time'] = \
        bilby.core.prior.Uniform(minimum=injection_parameters['geocent_time'] - 0.1, \
                                 maximum=injection_parameters['geocent_time'] + 0.1, \
                                 name='geocent_time', latex_label='$t_c$', unit='$s$')    
if zero_spins:
    to_fix += ['chi_1', 'chi_2']
else:
    priors.pop('chi_1')
    priors['chi_1'] = \
        bilby.gw.prior.AlignedSpin(a_prior=bilby.prior.Uniform(minimum=0, \
                                                               maximum=0.8), \
                                   z_prior=bilby.prior.Uniform(-1, 1), \
                                   name='chi_1', latex_label='$\chi_1$', \
                                   boundary='reflective')
for key in to_fix:
    priors[key] = injection_parameters[key]

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveoform generator, as well the priors.
# The explicit time, distance, and phase marginalizations are turned on to
# improve convergence, and the parameters are recovered by the conversion
# function.
if minimal_run:
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator,
        time_marginalization=False, phase_marginalization=False,
        distance_marginalization=False, priors=priors)
else:
    if pre_marge:
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
            distance_marginalization=True, phase_marginalization=True, 
            time_marginalization=True, jitter_time=True)
    else:
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
            distance_marginalization=False, phase_marginalization=False, 
            time_marginalization=False)
    
# Run sampler. In this case we're going to use the `cpnest` sampler
# Note that the maxmcmc parameter is increased so that between each iteration of
# the nested sampler approach, the walkers will move further using an mcmc
# approach, searching the full parameter space.
# The conversion function will determine the distance, phase and coalescence
# time posteriors in post processing.
if no_td:
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters
else:
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters
if sampler == 'emcee':
    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='emcee', nsteps=10000,
        injection_parameters=injection_parameters, outdir=outdir, label=label,
        conversion_function=conversion_function)
else:
    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000,
        injection_parameters=injection_parameters, outdir=outdir, label=label,
        conversion_function=conversion_function)

# Make a corner plot.
result.plot_corner()
