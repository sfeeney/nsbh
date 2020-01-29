import numpy as np
import bilby
import gwpy
import gwpy.signal
import matplotlib.pyplot as mp
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

def strain_plot(ifo, label=None, bandpass_frequencies=(50, 250), \
                notches=None, start_end=None, t0=None, ax=None, \
                ls='-', mark=None):
  
  # modified from Bilby's Interferometer.plot_time_domain_data()
  # function
  # https://git.ligo.org/lscsoft/bilby/blob/master/bilby/gw/detector/interferometer.py 
  
  # use the gwpy timeseries to perform bandpass and notching
  if notches is None:
      notches = list()
  timeseries = gwpy.timeseries.TimeSeries(
      data=ifo.strain_data.time_domain_strain, \
      times=ifo.strain_data.time_array)
  zpks = []
  if bandpass_frequencies is not None:
      zpks.append(gwpy.signal.filter_design.bandpass(
          bandpass_frequencies[0], bandpass_frequencies[1],
          ifo.strain_data.sampling_frequency))
  if notches is not None:
      for line in notches:
          zpks.append(gwpy.signal.filter_design.notch(
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


# common settings
npr_seed = 141023
outdir = 'outdir'
duration = 4.0 # 32.0
sampling_frequency = 2048.
minimum_frequency = 40.0 # 20.0
reference_frequency = 50.0
zero_spins = True
fig, axes = mp.subplots(2, 1, figsize=(16, 8), sharex=True)

# set up injected waveform
injection_parameters = dict(mass_1=7.39, mass_2=1.37, \
                            luminosity_distance=105.2307, \
                            theta_jn=0.4, psi=2.659, phase=1.3, \
                            geocent_time=1126259642.413, ra=1.375, \
                            dec=-1.2108, lambda_1=0, lambda_2=450)
if zero_spins:
    injection_parameters['chi_1'] = 0.0
    injection_parameters['chi_2'] = 0.0
else:
    injection_parameters['chi_1'] = 0.1
    injection_parameters['chi_2'] = 0.02
t0 = injection_parameters['geocent_time']
data_start_time = injection_parameters['geocent_time'] + 2 - duration

# instantiate injected waveform generator object
#waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', \
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2_NRTidal', \
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
strain_plot(H1, label='injected, noiseless', notches=[50], t0=t0, ax=axes[0])
H1 = bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
    'H1', injection_polarizations=hf_signal,
    injection_parameters=injection_parameters, duration=duration,
    sampling_frequency=sampling_frequency, start_time=data_start_time, 
    zero_noise=False, plot=False, save=False)
strain_plot(H1, label='injected', notches=[50], t0=t0, ax=axes[1])

# now determine maximum-likelihood parameters. read in 
# Bilby samples and find likelihood peak 
res_file = 'nsbh_inj_bilby_d_08.0_mf_40.0_rf_50.0_dynesty_zero_spin_result.json'
result = bilby.result.read_in_result(filename=outdir+'/'+res_file)
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
            ax=axes[0], ls=':', mark=injection_parameters['geocent_time'])
H1 = bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
    'H1', injection_polarizations=hf_signal,
    injection_parameters=injection_parameters, duration=duration,
    sampling_frequency=sampling_frequency, start_time=data_start_time, 
    zero_noise=False, plot=False, save=False)
strain_plot(H1, label='max like', notches=[50], t0=t0, ax=axes[1], ls=':', 
            mark=injection_parameters['geocent_time'])

# finish plot
fig.subplots_adjust(hspace=0, wspace=0)
for ax in axes:
  leg = ax.legend(loc='upper left', handlelength=5)
  for line in leg.get_lines():
    line.set_linewidth(lw)
mp.savefig(outdir + '/waveform_comparison.pdf', bbox_inches='tight')



