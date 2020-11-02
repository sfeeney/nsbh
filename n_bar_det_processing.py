import numpy as np
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import mpl_toolkits.axes_grid1 as mpag
import os.path as osp
import pickle

def d2z(d, h_0, q_0, order=3):

    z = h_0 * d / c
    if order > 1:
        z += -1.0 / 2.0 * (1.0 - q_0) * (h_0 * d / c) ** 2
    if order > 2:
        z += 1.0 / 6.0 * (4.0 - 7.0 * q_0 + 1.0) * (h_0 * d / c) ** 3

    return z


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
min_ejecta_mass = 0.01
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
store_all_selection = True

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

# calculate z_max used in simulations
z_max_fid = d2z(d_max, h_0_fid, q_0_fid)

# specific settings
n_grid = 5
n_procs = 96
n_runs = 0

# do we need to compile multiple files?
compile = True
if compile:

    # read in files
    data_list = []
    for i in range(n_procs):

        stub = '_n_det_proc_{:d}_of_{:d}.txt'.format(i, n_procs)
        data_file = osp.join(outdir, label + stub)
        data = np.genfromtxt(data_file, delimiter=',', names=True, dtype=None)
        data_list.append(data)
        n_runs += data.size

    # arrange into one useful array. bit of a faff.
    dtypes = data_list[0].dtype.fields
    data = np.empty((n_runs, ), dtype=dtypes)
    i_data = 0
    for i in range(n_procs):
        n_runs_i = data_list[i].size
        if n_runs_i == 1:
            data[i_data] = data_list[i]
            i_data += 1
        else:
            for j in range(n_runs_i):
                data[i_data] = data_list[i][j]
                i_data += 1

    # store processed data
    np.savetxt(osp.join(outdir, label + '_n_det.txt'), data, \
               delimiter=',', \
               header=','.join(data.dtype.names), \
               fmt='%d,%d,%d,%f,%f,%d,%d,%d,%f,%f')
    fname = osp.join(outdir, label + \
                     '_n_det_proc_*_of_{:d}.txt'.format(n_procs))
    print('results compiled. consider "rm ' + fname + '"')

else:

    # read in compiled results
    data_file = osp.join(outdir, label + '_n_det.txt')
    data = np.genfromtxt(data_file, delimiter=',', names=True, dtype=None)
    dtypes = data.dtype.fields
    n_runs = len(data['i_job'])

# summarize data
dtypes = {k: v for (k, v) in dtypes.items() if k != 'i_job'}
comp_data = np.zeros((n_grid ** 2, ), dtype=dtypes)
for i in range(n_runs):
    i_comp = data['i_h_0'][i] * n_grid + data['i_q_0'][i]
    comp_data['i_h_0'][i_comp] = data['i_h_0'][i]
    comp_data['i_q_0'][i_comp] = data['i_q_0'][i]
    comp_data['h_0'][i_comp] = data['h_0'][i]
    comp_data['q_0'][i_comp] = data['q_0'][i]

    comp_data['n_inj'][i_comp] += data['n_inj'][i]
    comp_data['n_det'][i_comp] += data['n_det'][i]
    comp_data['n_det_rem'][i_comp] += data['n_det_rem'][i]
    comp_data['z_max_det'][i_comp] = \
        max(comp_data['z_max_det'][i_comp], data['z_max_det'][i])
    comp_data['z_max_det_rem'][i_comp] = \
        max(comp_data['z_max_det_rem'][i_comp], data['z_max_det_rem'][i])
n_inj = np.reshape(comp_data['n_inj'], (n_grid, n_grid))
n_det = np.reshape(comp_data['n_det'], (n_grid, n_grid))
n_det_rem = np.reshape(comp_data['n_det_rem'], (n_grid, n_grid))
z_max_det = np.reshape(comp_data['z_max_det'], (n_grid, n_grid))
z_max_det_rem = np.reshape(comp_data['z_max_det_rem'], (n_grid, n_grid))
h_0_min = np.min(comp_data['h_0'])
h_0_max = np.max(comp_data['h_0'])
q_0_min = np.min(comp_data['q_0'])
q_0_max = np.max(comp_data['q_0'])

# plot
cmap = mpcm.plasma
worry = True
if worry:    
    cmap.set_bad(color='white')
    i_worry = np.abs(z_max_det / z_max_fid) > 0.999
    z_max_det[i_worry] = np.nan
    i_worry = np.abs(z_max_det_rem / z_max_fid) > 0.999
    z_max_det_rem[i_worry] = np.nan
to_plot = [n_inj, n_det, n_det_rem, z_max_det, z_max_det_rem]
titles = [r'$N_{\rm inj}$', r'$N_{\rm det}$', \
          r'$N_{\rm det,\,rem}$', r'$z^{\rm max}_{\rm det}$', \
          r'$z^{\rm max}_{\rm det,\,rem}$']
ticks = [0, 1, 2, 3, 4]
x_tick_labels = ['{:d}'.format(int(h_0)) for h_0 in \
                 np.linspace(h_0_min, h_0_max, n_grid)]
y_tick_labels = ['{:.2f}'.format(q_0) for q_0 in \
                 np.linspace(q_0_min, q_0_max, n_grid)]
fig, axes = mp.subplots(nrows=2, ncols=3, figsize=(10, 5))
n_cols = axes.shape[1]
for i in range(5):

    # find place in plot. skip one subplot.
    i_x = i // n_cols
    i_y = i % n_cols
    if i_x > 0:
        i_y += 1
    ax = axes[i_x, i_y]

    # plot
    im = ax.imshow(to_plot[i].T, interpolation='nearest', cmap=cmap)
    ax.set_xlabel(r'$H_0$')
    ax.set_ylabel(r'$q_0$')
    ax.set_title(titles[i])
    ax.set_xticks(ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(y_tick_labels)

    # colourbar shenanigans
    aspect = 20
    pad_fraction = 0.5
    divider = mpag.make_axes_locatable(ax)
    width = mpag.axes_size.AxesY(ax, aspect=1./aspect)
    pad = mpag.axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes('right', size=width, pad=pad)
    fig.colorbar(im, cax=cax)

# finish plot
fig.delaxes(axes[1, 0])
fig.subplots_adjust(wspace=0.8, hspace=0.4)
fig.savefig(osp.join(outdir, label + '_n_det.pdf'), bbox_inches='tight')
mp.close(fig)

# now plot redshifts vs SNR for detected events
if store_all_selection:
    fig, axes = mp.subplots(nrows=5, ncols=5, figsize=(10, 10), \
                            sharex=True, sharey=True)
    for i in range(n_procs):

        # read in selection file
        fname = 'data/' + label + \
                '_n_det_selection_proc_{:d}_of_{:d}.pkl'.format(i, n_procs)
        with open(fname, 'rb') as f:
            selection = pickle.load(f)

        # loop through contents, adding to plot
        for sel in selection:
            redshifts = sel[3]
            snrs = sel[4]
            ejecta_masses = sel[5]
            det = snrs > snr_thresh
            has_ejecta = ejecta_masses > min_ejecta_mass
            det_ej = np.logical_and(has_ejecta, det)
            #axes[sel[2], sel[1]].plot(redshifts[det], snrs[det], \
            #                          '.', color='C0')
            axes[sel[2], sel[1]].plot(redshifts[det_ej], snrs[det_ej], \
                                      '.', color='C0')

    # finish plot
    for i in range(n_grid):
        axes[4, i].set_xlabel('$z$')
        axes[i, 0].set_ylabel(r'$\rho$')
        for j in range(n_grid):
            axes[i, j].axhline(snr_thresh, ls='--', color='C1')
            par_label = r'$H_0=' + x_tick_labels[j] + '$'
            axes[i, j].text(0.95, 0.95, par_label, va='top', ha='right', \
                            transform=axes[i, j].transAxes)
            par_label = r'$q_0=' + y_tick_labels[i] + '$'
            axes[i, j].text(0.95, 0.85, par_label, va='top', ha='right', \
                            transform=axes[i, j].transAxes)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.savefig(osp.join(outdir, label + '_n_det_snrs.pdf'), \
                bbox_inches='tight')
