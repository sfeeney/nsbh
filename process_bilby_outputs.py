import numpy as np
import bilby
import gwpy
import gwpy.signal
import matplotlib.pyplot as mp
import copy
import os.path as osp

def strain_plot(ifo, label=None, bandpass_frequencies=(50, 250), \
                notches=None, start_end=None, t0=None, ax=None, \
                ls='-', mark=None):
  
    # modified from Bilby's Interferometer.plot_time_domain_data()
    # function
    # https://git.ligo.org/lscsoft/bilby/blob/master/bilby/gw/detector/interferometer.py 

    # use the gwpy timeseries to perform bandpass and notching
    if notches is None:
        notches = list()
    timeseries = \
        gwpy.timeseries.TimeSeries(data=ifo.strain_data.time_domain_strain, \
                                   times=ifo.strain_data.time_array)
    zpks = []
    if bandpass_frequencies is not None:
        zpks.append(gwpy.signal.filter_design.bandpass( \
            bandpass_frequencies[0], bandpass_frequencies[1], \
            ifo.strain_data.sampling_frequency))
    if notches is not None:
        for line in notches:
            zpks.append(gwpy.signal.filter_design.notch( \
                        line, ifo.strain_data.sampling_frequency))
    if len(zpks) > 0:
        zpk = gwpy.signal.filter_design.concatenate_zpks(*zpks)
        strain = timeseries.filter(zpk, filtfilt=False)
    else:
        strain = timeseries

    # check for exisiting axes and plot
    if ax is None:
        fig, ax = mp.subplots(figsize=(16, 5))
    if t0:
        x = ifo.strain_data.time_array - t0
        xlabel = 'GPS time [s] - {}'.format(t0)
        if mark is not None:
            ax.axvline(mark - t0, color='Gray', ls='--')
    else:
        x = ifo.strain_data.time_array
        xlabel = 'GPS time [s]'
        if mark is not None:
            ax.axvline(mark, color='Gray', ls='--')
    if label is not None:
        ax.plot(x, strain, label=label, ls=ls)
    else:
        ax.plot(x, strain, ls=ls)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Strain')
    if start_end is not None:
        ax.set_xlim(*start_end)
    return ax


# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# common settings
fig, axes = mp.subplots(2, 1, figsize=(16, 8), sharex=True)
npr_seed = 141023
duration = 8.0 # 32.0
sampling_frequency = 2048.
minimum_frequency = 40.0 # 20.0
reference_frequency = 50.0
sampler = 'dynesty' # 'emcee'
zero_spins = True
minimal_run = False
pre_marge = True
no_td = False
plot_start_end = [-3, 1]
outdir = 'outdir'
label_str = 'nsbh_inj_bilby_d_{:04.1f}_mf_{:4.1f}_rf_{:4.1f}_{:s}'
label = label_str.format(duration, minimum_frequency, \
                         reference_frequency, sampler)
if zero_spins:
    label += '_zero_spin'
if minimal_run:
    label += '_mini'
else:
    if pre_marge:
        label += '_dpt_marge'
if no_td:
    label += '_no_td'

# read in results file, which contains tonnes of info 
res_file = label + '_result.json'
result = bilby.result.read_in_result(filename=osp.join(outdir, res_file))
truths = copy.deepcopy(result.injection_parameters)

# set up injected waveform
injected_pars = ['mass_1', 'mass_2', 'luminosity_distance', \
                 'theta_jn', 'psi', 'phase', 'geocent_time', \
                 'ra', 'dec', 'chi_1', 'chi_2']
if not no_td:
    injected_pars += ['lambda_1', 'lambda_2']
injection_parameters = {key: truths[key] for key in injected_pars}
t0 = truths['geocent_time']
data_start_time = truths['geocent_time'] + 2 - duration

# instantiate injected waveform generator object
if no_td:
    wf_approx = 'IMRPhenomPv2'
else:
    wf_approx = 'IMRPhenomPv2_NRTidal'
waveform_arguments = dict(waveform_approximant=wf_approx, \
                          reference_frequency=reference_frequency, \
                          minimum_frequency=minimum_frequency)
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments)

# generate noiseless and noisy signals in single detector, and plot
#injection_parameters['luminosity_distance'] = 1.0e6
hf_signal = waveform_generator.frequency_domain_strain(injection_parameters)
np.random.seed(npr_seed)
H1 = bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
    'H1', injection_polarizations=hf_signal,
    injection_parameters=injection_parameters, duration=duration,
    sampling_frequency=sampling_frequency, start_time=data_start_time, 
    zero_noise=True, plot=False, save=False)
strain_plot(H1, label='injected, noiseless', notches=[50], t0=t0, \
            ax=axes[0], start_end=plot_start_end)
H1 = bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
    'H1', injection_polarizations=hf_signal,
    injection_parameters=injection_parameters, duration=duration,
    sampling_frequency=sampling_frequency, start_time=data_start_time, 
    zero_noise=False, plot=False, save=False)
strain_plot(H1, label='injected', notches=[50], t0=t0, ax=axes[1], \
            start_end=plot_start_end)

# now determine maximum-likelihood parameters
i_ml = result.posterior['log_likelihood'].idxmax
ml = result.posterior.loc[i_ml]
for key in injection_parameters.keys():
    injection_parameters[key] = ml[key]

# generate signal in single detector
#injection_parameters['luminosity_distance'] = 1.0e6
hf_signal = waveform_generator.frequency_domain_strain(injection_parameters)
np.random.seed(npr_seed)
H1 = bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
    'H1', injection_polarizations=hf_signal,
    injection_parameters=injection_parameters, duration=duration,
    sampling_frequency=sampling_frequency, start_time=data_start_time, 
    zero_noise=True, plot=False, save=False)
strain_plot(H1, label='max like, noiseless', notches=[50], t0=t0, \
            ax=axes[0], ls=':', mark=injection_parameters['geocent_time'], \
            start_end=plot_start_end)
H1 = bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
    'H1', injection_polarizations=hf_signal,
    injection_parameters=injection_parameters, duration=duration,
    sampling_frequency=sampling_frequency, start_time=data_start_time, 
    zero_noise=False, plot=False, save=False)
strain_plot(H1, label='max like', notches=[50], t0=t0, ax=axes[1], ls=':', 
            mark=injection_parameters['geocent_time'], \
            start_end=plot_start_end)

# finish plot
fig.subplots_adjust(hspace=0, wspace=0)
for ax in axes:
    leg = ax.legend(loc='upper left', handlelength=5)
    for line in leg.get_lines():
        line.set_linewidth(lw)
fig.savefig(osp.join(outdir, label + '_wf_in_out.pdf'), \
            bbox_inches='tight')

# corner plot including luminosity distance
if zero_spins:

    plot_params = {key: result.injection_parameters[key] for key in \
                   result.search_parameter_keys}
    if no_td:
        plot_params.pop('time_jitter')
        plot_params['luminosity_distance'] = \
            result.injection_parameters['luminosity_distance']
    fig = result.plot_corner(parameters=plot_params, save=False)
    if no_td:
        n = len(plot_params.keys())
        fig.axes[n ** 2 - n].set_ylabel(r'$d_L\,{\rm [Mpc]}$')
        fig.axes[n ** 2 - 1].set_xlabel(r'$d_L\,{\rm [Mpc]}$')
    fig.savefig(osp.join(outdir, label + '_final_corner.pdf'), \
                bbox_inches='tight')

# @TODO
# non-zero-spin run with marginalized likes!
# email gang, nikhil
# filename


