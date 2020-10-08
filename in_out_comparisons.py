import numpy as np
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import bilby
import bilby.gw.conversion as bc
import getdist as gd
import getdist.plots as gdp
import os.path as osp
import ns_eos_aw as nseos

def comp_masses_to_chirp_q(m_1, m_2):

    m_c = (m_1 * m_2) ** 0.6 / (m_1 + m_2) ** 0.2
    q_inv = m_2 / m_1

    return m_c, q_inv

def m_rem(m_1, a_1_z, m_2):

    sim = {'mass1': m_1, 'spin1x': 0.0, \
           'spin1y': 0.0, 'spin1z': a_1_z, \
           'mass2': m_2, 'spin2x': 0.0, \
           'spin2y': 0.0, 'spin2z': 0.0}
    
    return nseos.Foucart(sim, eos="DD2").remnant_mass()

def bisection(a_min, a_max, m_1, m_2):

    # simple fixed number of steps for now
    for i in range(20):

        # guess new spin between current min and max
        a_guess = (a_min + a_max) / 2.0
        m_rem_guess = m_rem(m_1, a_guess, m_2)

        # if guess produces remnant with mass above minimum, 
        # set guessed spin to new maximum; else, set to new 
        # minimum
        if m_rem_guess > min_mass:
            a_max = a_guess
        else:
            a_min = a_guess

    return a_guess

# plot settings
lw = 1.5
mp.rc('font', family='serif', size=10)
mp.rcParams['text.usetex'] = True
#mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# settings
proposal_version = True
min_network = False
if min_network:
    ifo_list = ['H1', 'L1', 'V1', 'K1-']
    t_obs = 3.0 # years
else:
    ifo_list = ['H1+', 'L1+', 'V1+', 'K1+', 'A1']
    t_obs = 5.0 # years
sample_z = True
redshift_rate = True
uniform_bh_masses = True
uniform_ns_masses = True
low_metals = True
broad_bh_spins = True
zero_spins = False
tight_loc = False
fixed_ang = True
sample_z = True
seobnr_waveform = True
if seobnr_waveform:
    waveform_approximant = 'SEOBNRv4_ROM_NRTidalv2_NSBH'
    aligned_spins = True
    use_polychord = True
    n_live = 1000
else:
    waveform_approximant = 'IMRPhenomPv2_NRTidal'
    aligned_spins = False
    use_polychord = False
    n_live = 1000
use_weighted_samples = False
min_mass = 0.01

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
datdir = 'data'
outdir = 'outdir'

# getdist settings
m_c_min, _ = comp_masses_to_chirp_q(m_min_bh, m_min_ns)
m_c_max, _ = comp_masses_to_chirp_q(m_max_bh, m_max_ns)
_, q_inv_min = comp_masses_to_chirp_q(m_max_bh, m_min_ns)
_, q_inv_max = comp_masses_to_chirp_q(m_min_bh, m_max_ns)
gd_ranges={'a_1':(spin_min_bh, spin_max_bh), \
           'iota':(0.0, np.pi), \
           'q':(q_inv_min, q_inv_max), \
           'lambda_2':(0.0, 4500.0), \
           'lambda_tilde':(0.0, None)}

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
base_label = label_str.format(duration, minimum_frequency, \
                              reference_frequency)

# quick bisection search for disruption line as a function of 
# black hole mass. do so for min, mean and max neutron star 
# mass
n_grid = 100
m_1_grid = np.linspace(m_min_bh, m_max_bh, n_grid)
if uniform_ns_masses:
    m_mean_ns = (m_max_ns + m_min_ns) / 2.0
a_min_m_ns_mean = np.zeros(n_grid)
a_min_m_ns_min = np.zeros(n_grid)
a_min_m_ns_max = np.zeros(n_grid)
for i in range(n_grid):

    a_min_m_ns_mean[i] = bisection(1.0e-10, spin_max_bh, m_1_grid[i], \
                                   m_mean_ns)
    a_min_m_ns_min[i] = bisection(1.0e-10, spin_max_bh, m_1_grid[i], \
                                  m_min_ns)
    a_min_m_ns_max[i] = bisection(1.0e-10, spin_max_bh, m_1_grid[i], \
                                  m_max_ns)

# read in injections from file
par_file = datdir + '/' + base_label + '.txt'
pars = np.genfromtxt(par_file, dtype=None, names=True, \
                     delimiter=',', encoding=None)
det = pars['snr'] >= snr_thresh
rem = pars['remnant_mass'] > min_mass
det_rem = np.logical_and(det, rem)
not_det_not_rem = np.logical_and(~det, ~rem)
n_inj = len(pars)
n_det = np.sum(det)
n_rem = np.sum(rem)
n_det_rem = np.sum(det_rem)
if seobnr_waveform:
    a_mag_1 = pars['spin1z']
else:
    a_mag_1 = np.sqrt(pars['spin1x'] ** 2 + \
                      pars['spin1y'] ** 2 + \
                      pars['spin1z'] ** 2)

# plot population draws
if proposal_version:
    fig, axes = mp.subplots(nrows=1, ncols=2, figsize=(10, 3.125))
else:
    fig, axes = mp.subplots(nrows=1, ncols=2, figsize=(10, 5))

# plot
col_u = 'C6' # 'C4' 'C1'
col_d = 'C0'
min_mass_text = '{:.2f}'.format(min_mass)
#if proposal_version:
if False:
    axes[0].scatter(pars['mass1'][det], a_mag_1[det], marker='o', s=15, \
                    c=col_u, edgecolors=col_u, alpha=0.75, \
                    label=r'detected, no ejecta', rasterized=True)
    axes[0].scatter(pars['mass1'][det_rem], a_mag_1[det_rem], \
                    marker='o', s=15, c=col_d, edgecolors=col_d, \
                    label=r'detected, ejecta', rasterized=True)
else:
    axes[0].scatter(pars['mass1'][not_det_not_rem], \
                    a_mag_1[not_det_not_rem], marker='o', s=15, \
                    c='none', edgecolors=col_u, alpha=0.25, \
                    label=r'undetected, no ejecta', rasterized=True)
    axes[0].scatter(pars['mass1'][rem], a_mag_1[rem], marker='o', s=15, \
                    c='none', edgecolors=col_d, alpha=0.25, \
                    label=r'undetected, ejecta', rasterized=True)
    axes[0].scatter(pars['mass1'][det], a_mag_1[det], marker='o', s=15, \
                    c=col_u, edgecolors='k', alpha=0.75, \
                    label=r'detected, no ejecta', rasterized=True)
    axes[0].scatter(pars['mass1'][det_rem], a_mag_1[det_rem], \
                    marker='o', s=15, c=col_d, edgecolors='k', \
                    label=r'detected, ejecta', rasterized=True)

# remove unecessary axis labels and ticks
axes[0].tick_params(axis='both', which='major', labelsize=12)
if proposal_version:
    axes[0].set_xlabel(r'BH mass ($M_\odot$)', fontsize=12)
    axes[0].set_ylabel(r'BH spin magnitude', fontsize=12)
else:
    axes[0].set_xlabel(r'$M_{\rm BH}$', fontsize=12)
    axes[0].set_ylabel(r'$|a_{\rm BH}|$', fontsize=12)

# customized legend for each plot
alt_legend = False
if alt_legend:
    legend = '{:d} detected'.format(n_det) + \
             '\n{:d} w/ ejecta'.format(n_rem) + \
             '\n{:d} det w/ ejecta'.format(n_det_rem)
    axes[0].text(0.95, 0.05, legend, transform=axes[0].transAxes, \
                    fontsize=11, va='bottom', ha='right', \
                    bbox={'fc': 'white', 'ec':'black'})
else:
    axes[0].legend(loc='lower right', fontsize=11, handlelength=1.8, \
                   bbox_to_anchor=(0.98, 0.02), alpha=1.0)
    leg = axes[0].get_legend()
    leg.get_frame().set_edgecolor('k')
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

# rescale if desired
axes[0].grid(False)
axes[0].set_xlim(m_min_bh, m_max_bh)
axes[0].set_ylim(spin_min_bh, spin_max_bh)

# overlay estimates of disruption line
col_drl = 'k'
axes[0].plot(m_1_grid, a_min_m_ns_min, color=col_drl, ls='-.')
axes[0].plot(m_1_grid, a_min_m_ns_mean, color=col_drl)
axes[0].plot(m_1_grid, a_min_m_ns_max, color=col_drl, ls='--')

# pick out detections
pars = pars[det_rem]
ids = np.array([int(i_sim.split(':')[-1]) for i_sim in \
                pars['simulation_id']])
snrs = pars['snr']
i_sort = np.argsort(snrs)[::-1]
target_snrs = snrs[i_sort]
target_ids = ids[i_sort]

# loop over targets
n_targets = len(target_ids)
samples = []
truths = []
skip = np.full(n_targets, False)
#n_targets = 10
for i in range(n_targets):

    # read in results file, which contains tonnes of info
    label = base_label + '_inj_{:d}'.format(target_ids[i])
    if use_polychord:
        label = 'pc_' + label
    if zero_spins:
        label += '_zero_spins'
    if tight_loc:
        label += '_tight_loc'
    elif fixed_ang:
        label += '_fixed_ang'
    if n_live != 1000:
        label += '_nlive_{:04d}'.format(n_live)
    res_file = label + '_result.json'
    print(osp.join(outdir, res_file))
    if not osp.exists(osp.join(outdir, res_file)):
        skip[i] = True
        samples.append(None)
        truths.append(None)
        continue
    result = bilby.result.read_in_result(filename=osp.join(outdir, res_file))
    all_pars = bc.generate_all_bns_parameters(result.injection_parameters)
    truths.append([all_pars['mass_1'], \
                   all_pars['a_1'], 0.0, \
                   all_pars['iota'], \
                   all_pars['luminosity_distance'], \
                   all_pars['mass_ratio'], \
                   all_pars['lambda_2'], \
                   all_pars['lambda_tilde'], \
                   all_pars['spin_1z']])

    # if sampling with polychord optionally use full, variable weight 
    # posterior samples: bilby takes the equal-weight posterior samples 
    # to build its result.posterior. there are no distance samples this
    # way though!
    if use_polychord and use_weighted_samples:

        # set up required paramnames file
        if aligned_spins:
            template = 'nsbh_aligned_spins.paramnames'
        else:
            template = 'nsbh_precess_spins.paramnames'
        gd_root = osp.join(outdir, osp.join('chains', label))
        gd_pars = gd_root + '.paramnames'
        if not osp.exists(gd_pars):
            os.symlink(template, gd_pars)

        # read in samples and fill in derived parameters
        gd_samples = gd.loadMCSamples(gd_root)
        pars = gd_samples.getParams()
        m_1, m_2 = chirp_q_to_comp_masses(pars.chirp_mass, \
                                          pars.mass_ratio)
        gd_samples.addDerived(m_1, name='mass_1', label='m_1')
        gd_samples.addDerived(m_2, name='mass_2', label='m_2')
        gd_samples.addDerived(np.abs(pars.chi_1), name='a_1', label='a_1')
        samples.append(gd_samples)

    else:
        
        # convert to GetDist MCSamples object
        distance_label = r'(d_L - d_L^{\rm true})/d_L^{\rm true}'
        try:
            delta_distance = \
                (result.posterior.luminosity_distance - \
                 result.injection_parameters['luminosity_distance']) / \
                result.injection_parameters['luminosity_distance']
        except ValueError:
            skip[i] = True
            samples.append(None)
            continue
        if aligned_spins:
            gd_samples = np.array([result.posterior.a_1, \
                                   result.posterior.mass_1, \
                                   delta_distance, \
                                   result.posterior.iota, \
                                   result.posterior.mass_ratio, \
                                   result.posterior.lambda_2, \
                                   result.posterior.lambda_tilde, \
                                   result.posterior.chi_1]).T
            samples.append(gd.MCSamples(samples=gd_samples, \
                                        names=['a_1', 'mass_1', \
                                               'distance', 'iota', \
                                               'q', 'lambda_2', \
                                               'lambda_tilde', 'chi_1'], \
                                        labels=['|a|_1', r'M_{\rm BH}', \
                                                distance_label, r'\iota', \
                                                'm_2/m_1', r'\Lambda_{\rm NS}', \
                                                r'\tilde{\Lambda}', r'\chi_1'], \
                                        ranges=gd_ranges))
        else:
            gd_samples = np.array([result.posterior.a_1, \
                                   result.posterior.mass_1, \
                                   delta_distance, \
                                   result.posterior.iota, \
                                   result.posterior.mass_ratio, \
                                   result.posterior.lambda_2, \
                                   result.posterior.lambda_tilde]).T
            samples.append(gd.MCSamples(samples=gd_samples, \
                                        names=['a_1', 'mass_1', \
                                               'distance', 'iota', \
                                               'q', 'lambda_2', \
                                               'lambda_tilde'], \
                                        labels=['|a|_1', r'M_{\rm BH}', \
                                                distance_label, r'\iota', \
                                                'm_2/m_1', r'\Lambda_{\rm NS}', \
                                                r'\tilde{\Lambda}'], \
                                        ranges=gd_ranges))

# snr-ordered colouring
n_targets = len(samples)
#cm = mpcm.get_cmap('Oranges')
cm = mpcm.get_cmap('Blues')
cols = [mpc.rgb2hex(cm(x)) for x in np.linspace(0.2, 0.8, n_targets)]

# tweak output filename
if use_polychord:
    base_label = 'pc_' + base_label

# single-axis BH mass-spin plot
g = gdp.get_single_plotter()
g.settings.num_plot_contours = 1
g.settings.alpha_factor_contour_lines = 0
scatter = False
for i in range(n_targets - 1, -1, -1):
    if skip[i]:
        continue
    else:
        col = cols[n_targets - 1 - i]
        #col = cols[i]
        if scatter:
            g.plot_2d_scatter(samples[i], 'mass_1', 'a_1', color=col, \
                              ax=axes[1], alpha=0.05)
        else:
            alpha = 0.8 # 0.5
            samples[i].updateSettings({'contours': [0.95]})
            g.plot_2d(samples[i], 'mass_1', 'a_1', colors=[col], \
                      ax=axes[1], filled=True, alphas=[alpha])
            #axes[1].plot([truths[i][0]], [truths[i][1]], \
            #             marker='+', color=col)
axes[1].grid(False)
axes[1].set_xlim(m_min_bh, m_max_bh)
axes[1].set_ylim(spin_min_bh, spin_max_bh)
axes[1].get_yaxis().set_visible(False)
axes[1].tick_params(axis='both', which='major', labelsize=12)
if proposal_version:
    axes[1].set_xlabel(r'BH mass ($M_\odot$)', fontsize=12)
else:
    axes[1].set_xlabel(r'$M_{\rm BH}$', fontsize=12)

# overlay estimates of disruption line
col_drl = 'k'
axes[1].plot(m_1_grid, a_min_m_ns_min, color=col_drl, ls='-.', \
             label='NS mass: {:.2f}'.format(m_min_ns) + r' $M_\odot$')
axes[1].plot(m_1_grid, a_min_m_ns_mean, color=col_drl, \
             label='NS mass: {:.2f}'.format(m_mean_ns) + r' $M_\odot$')
axes[1].plot(m_1_grid, a_min_m_ns_max, color=col_drl, ls='--', \
             label='NS mass: {:.2f}'.format(m_max_ns) + r' $M_\odot$')
axes[1].legend(loc='lower right', fontsize=11, handlelength=1.8, \
               bbox_to_anchor=(0.98, 0.02), \
               title='tidal disruption lines')
leg = axes[1].get_legend()
leg.get_frame().set_edgecolor('k')
for line in leg.get_lines():
    line.set_linewidth(lw)

# finish plot
axes[0].set_title('Simulation Properties')
axes[1].set_title('Recovered Properties')
fig.subplots_adjust(wspace=0.0, hspace=0.0)
if scatter:
    fig.savefig(osp.join(outdir, base_label + \
                         '_disruption_line_in_out_comp_scatter.pdf'), \
                bbox_inches='tight', dpi=300)
else:
    fig.savefig(osp.join(outdir, base_label + '_disruption_line_in_out_comp.pdf'), \
                bbox_inches='tight', dpi=300)

