import numpy as np
import bilby
import bilby.gw.conversion as bc
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import os as os
import os.path as osp
import getdist as gd
import getdist.plots as gdp
import copy
#import matplotlib
#matplotlib.use('TkAgg')

def comp_masses_to_chirp_q(m_1, m_2):

    m_c = (m_1 * m_2) ** 0.6 / (m_1 + m_2) ** 0.2
    q_inv = m_2 / m_1

    return m_c, q_inv

def chirp_q_to_comp_masses(m_c, q_inv):

    q = 1.0 / q_inv
    m_2 = (1 + q) ** 0.2 / q ** 0.6 * m_c
    m_1 = q * m_2

    return m_1, m_2

def prior_change_jac(m_c, q_inv):

    d_m_1_d_m_c = (1.0 + q_inv) ** 0.2 / q_inv ** 0.6
    d_m_2_d_m_c = (1.0 + q_inv) ** 0.2 * q_inv ** 0.4
    d_m_1_d_q_inv = -(3.0 + 2.0 * q_inv) * m_c / 5.0 / \
                    q_inv ** 1.6 * (1.0 + q_inv) ** 0.8
    d_m_2_d_q_inv = (2.0 + 3.0 * q_inv) * m_c / 5.0 / \
                    q_inv ** 0.6 * (1.0 + q_inv) ** 0.8
    jac = np.abs(d_m_1_d_m_c * d_m_2_d_q_inv - \
                 d_m_1_d_q_inv * d_m_2_d_m_c)
    return jac

# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# settings
use_mpi = False
snr_thresh = 12.0
duration = 32.0 # 8.0
sampling_frequency = 2048.
minimum_frequency = 20.0 # 40.0
reference_frequency = 14.0 # 50.0
min_network = False
if min_network:
    ifo_list = ['H1', 'L1', 'V1', 'K1-']
else:
    ifo_list = ['H1+', 'L1+', 'V1+', 'K1+', 'A1']
use_polychord = True
use_weighted_samples = False
imp_sample = True
if use_polychord:
    n_live = 1000 # 500
else:
    n_live = 1000
zero_spins = False
remnants_only = True
min_remnant_mass = 0.01
tight_loc = False
fixed_ang = True
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
lam_det_test = False
old = False
outdir = 'outdir'

# BH mass and spin prior limits
if uniform_bh_masses:
    m_min_bh = 2.5
    if low_metals:
        m_max_bh = 40.0
    else:
        m_max_bh = 12.0
else:
    m_min_bh = 5.0
    m_max_bh = 20.0
spin_min_bh = 0.0
if broad_bh_spins:
    spin_max_bh = 0.99
else:
    spin_max_bh = 0.5

# NS mass and spin prior limits
m_min_ns = 1.0
if uniform_ns_masses:
    m_max_ns = 2.42
else:
    m_max_ns = 2.0
spin_min_ns = 0.0
spin_max_ns = 0.05

# conversions
m_c_min, _ = comp_masses_to_chirp_q(m_min_bh, m_min_ns)
m_c_max, _ = comp_masses_to_chirp_q(m_max_bh, m_max_ns)
_, q_inv_min = comp_masses_to_chirp_q(m_max_bh, m_min_ns)
_, q_inv_max = comp_masses_to_chirp_q(m_min_bh, m_max_ns)

# getdist settings
if old:
    gd_ranges={'a_1':(0.0, 0.8), 'iota':(0.0, np.pi), \
               'q':(0.02, 0.4), 'lambda_2':(0.0, 4000.0), \
               'lambda_tilde':(0.0, None)}
else:
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

# read injections from file
if old:

    targets = np.genfromtxt('data/remnant_sorted_detected.txt', delimiter=' ')
    target_ids = targets[:, 0].astype(int)
    target_snrs = targets[:, 2]
    if remnants_only:
        target_ids = target_ids[targets[:, 1] > 0.0]
        target_snrs = target_snrs[targets[:, 1] > 0.0]

else:

    par_file = base_label + '.txt'
    raw_pars = np.genfromtxt('data/' + par_file, \
                             dtype=None, names=True, delimiter=',', \
                             encoding=None)
    det = raw_pars['snr'] >= snr_thresh
    if remnants_only:
        det = np.logical_and(det, raw_pars['remnant_mass'] > min_remnant_mass)
    raw_pars = raw_pars[det]
    ids = np.array([int(i_sim.split(':')[-1]) for i_sim in \
                    raw_pars['simulation_id']])
    snrs = raw_pars['snr']
    i_sort = np.argsort(snrs)[::-1]
    target_snrs = snrs[i_sort]
    target_ids = ids[i_sort]
    target_iotas = raw_pars['inclination'][i_sort]

# loop over targets
n_targets = len(target_ids)
samples = []
truths = []
skip = np.full(n_targets, False)
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

    # NB: result.injection_parameters contains incorrect IMRPhenom iotas
    # due to a bug in bilby!
    all_pars = bc.generate_all_bns_parameters(result.injection_parameters)
    truths.append([all_pars['mass_1'], \
                   all_pars['a_1'], 0.0, \
                   target_iotas[i], \
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
        print(gd_pars)
        if not osp.exists(gd_pars):
            os.symlink(template, gd_pars)

        # read in samples and fill in derived parameters
        gd_samples = gd.loadMCSamples(gd_root)
        pars = gd_samples.getParams()
        m_1, m_2 = chirp_q_to_comp_masses(pars.chirp_mass, \
                                          pars.mass_ratio)
        gd_samples.addDerived(m_1, name='mass_1', label='m_1')
        gd_samples.addDerived(m_2, name='mass_2', label='m_2')
        if aligned_spins:
            gd_samples.addDerived(np.abs(pars.chi_1), name='a_1', label='a_1')

        # optionally importance sample the input mass priors
        if imp_sample:

            # extract posterior samples relevant for reweighting
            m_c_samples = pars.chirp_mass
            q_inv_samples = pars.mass_ratio
            m_1_samples, m_2_samples = \
                chirp_q_to_comp_masses(m_c_samples, q_inv_samples)

            # define importance weights. we want to convert from the 
            # prior we used to sample, which is uniform in chirp mass 
            # and mass to the prior we used to simulate, which is 
            # uniform in component masses. we technically used a different
            # redshift prior too, but they're almost identical in practice.
            # note that our sampling prior extends into regions where the 
            # component-mass prior is zero: apply tiny weights to these 
            # values
            mass_weights = prior_change_jac(m_c_samples, q_inv_samples)
            m_1_mask = np.logical_and(m_1_samples >= m_min_bh, \
                                      m_1_samples <= m_max_bh)
            m_2_mask = np.logical_and(m_2_samples >= m_min_ns, \
                                      m_2_samples <= m_max_ns)
            m12_mask = np.logical_and(m_1_mask, m_2_mask)
            mass_weights *= m12_mask
            mass_weights += ~m12_mask * 1.0e-10
            weights = mass_weights / np.sum(mass_weights)
            gd_samples.reweightAddingLogLikes(-np.log(weights))

        # build up list of samples objects
        samples.append(gd_samples)

    else:
        
        # test posterior is correctly sampled
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

        # optionally importance sample the input mass priors
        if imp_sample:

            # extract posterior samples relevant for reweighting
            m_c_samples = result.posterior.chirp_mass
            q_inv_samples = result.posterior.mass_ratio
            m_1_samples, m_2_samples = \
                chirp_q_to_comp_masses(m_c_samples, q_inv_samples)

            # define importance weights. we want to convert from the 
            # prior we used to sample, which is uniform in chirp mass 
            # and mass to the prior we used to simulate, which is 
            # uniform in component masses. we technically used a different
            # redshift prior too, but they're almost identical in practice.
            # note that our sampling prior extends into regions where the 
            # component-mass prior is zero: apply tiny weights to these 
            # values
            mass_weights = prior_change_jac(m_c_samples, q_inv_samples)
            m_1_mask = np.logical_and(m_1_samples >= m_min_bh, \
                                      m_1_samples <= m_max_bh)
            m_2_mask = np.logical_and(m_2_samples >= m_min_ns, \
                                      m_2_samples <= m_max_ns)
            m12_mask = np.logical_and(m_1_mask, m_2_mask)
            mass_weights *= m12_mask
            mass_weights += ~m12_mask * 1.0e-10
            weights = mass_weights / np.sum(mass_weights)

        else:

            weights = np.ones(len(result.posterior.luminosity_distance))
            weights /= np.sum(weights)

        # convert to GetDist MCSamples object
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
                                        labels=['a_1', 'm_1', \
                                                distance_label, r'\iota', \
                                                'm_2/m_1', r'\Lambda_{\rm NS}', \
                                                r'\tilde{\Lambda}', r'\chi_1'], \
                                        ranges=gd_ranges, weights=weights))
        else:
            gd_samples = np.array([result.posterior.a_1, \
                                   result.posterior.mass_1, \
                                   delta_distance, \
                                   result.posterior.iota, \
                                   result.posterior.mass_ratio, \
                                   result.posterior.lambda_2, \
                                   result.posterior.lambda_tilde, \
                                   result.posterior.spin_1z]).T
            samples.append(gd.MCSamples(samples=gd_samples, \
                                        names=['a_1', 'mass_1', \
                                               'distance', 'iota', \
                                               'q', 'lambda_2', \
                                               'lambda_tilde', 'chi_1'], \
                                        labels=['a_1', 'm_1', \
                                                distance_label, r'\iota', \
                                                'm_2/m_1', r'\Lambda_{\rm NS}', \
                                                r'\tilde{\Lambda}', r'\chi_1'], \
                                        ranges=gd_ranges, weights=weights))

# snr-ordered colouring
n_targets = len(samples)
cm = mpcm.get_cmap('plasma')
cols = [mpc.rgb2hex(cm(x)) for x in np.linspace(0.2, 0.8, n_targets)[::-1]]

# tweak output filename
if use_polychord:
    base_label = 'pc_' + base_label
if imp_sample:
    base_label = base_label + '_imp_sample'
if use_polychord and use_weighted_samples:
    base_label = base_label + '_weighted_samples'

# single-axis BH mass-spin plot
fig, ax = mp.subplots()
g = gdp.get_single_plotter()
scatter = True
for i in range(n_targets - 1, -1, -1):
    if skip[i]:
        continue
    else:
        col = cols[n_targets - 1 - i]
        #col = cols[i]
        if scatter:
            g.plot_2d_scatter(samples[i], 'mass_1', 'a_1', color=col, \
                              ax=ax, alpha=0.05)
        else:
            g.plot_2d(samples[i], 'mass_1', 'a_1', colors=[col], \
                      ax=ax, filled=True)
            ax.plot([truths[i][0]], [truths[i][1]], \
                    marker='+', color=col)
ax.grid(False)
ax.set_xlim(m_min_bh, m_max_bh)
ax.set_ylim(spin_min_bh, spin_max_bh)
fig.subplots_adjust(wspace=0.0, hspace=0.0)
if scatter:
    fig.savefig(osp.join(outdir, base_label + \
                         '_disruption_line_scatter.pdf'), \
                bbox_inches='tight')
else:
    fig.savefig(osp.join(outdir, base_label + '_disruption_line.pdf'), \
                bbox_inches='tight')

# generate figure and plot!
n_col = 8
n_row = int(np.ceil(n_targets / float(n_col)))
height = 2.86 * n_row
n_ext = n_col * n_row - n_targets
fig, axes = mp.subplots(n_row, n_col, figsize=(20, height))
g = gdp.get_single_plotter()
i_x = 0
i_y = 0
for i in range(n_row * n_col):
    
    if i < n_targets:

        if skip[i]:
            fig.delaxes(axes[i_y, i_x])
            i_x += 1
            if i_x == axes.shape[1]:
                i_x = 0
                i_y += 1
            continue

        # plot mass and spin constraints
        #if not aligned_spins:
        if aligned_spins:
            g.plot_2d(samples[i], 'mass_1', 'chi_1', colors=[cols[i]], \
                      ax=axes[i_y, i_x], filled=True)
            axes[i_y, i_x].plot([truths[i][0]], [truths[i][8]], \
                                marker='+', color='k')
        else:
            g.plot_2d(samples[i], 'mass_1', 'a_1', colors=[cols[i]], \
                      ax=axes[i_y, i_x], filled=True)
            axes[i_y, i_x].plot([truths[i][0]], [truths[i][1]], \
                                marker='+', color='k')
        label = r'$\rho=' + '{:4.1f}'.format(target_snrs[i]) + '$'
        axes[i_y, i_x].text(0.95, 0.95, label, ha='right', va='top', \
                            transform=axes[i_y, i_x].transAxes)
        axes[i_y, i_x].grid(False)
        axes[i_y, i_x].set_xlim(m_min_bh, 23.0) #m_max_bh)
        if aligned_spins:
            axes[i_y, i_x].set_ylim(-spin_max_bh, spin_max_bh)
        else:
            #axes[i_y, i_x].set_ylim(spin_min_bh, spin_max_bh)
            #axes[i_y, i_x].set_yticks([0.1, 0.3, 0.5, 0.7])
            axes[i_y, i_x].set_ylim(0.1, spin_max_bh)

        # remove axis labels where they would otherwise overlap
        if i_x > 0:
            axes[i_y, i_x].get_yaxis().set_visible(False)
        if i_y < n_row - 1:
            if i_y != n_row - 2 or i_x <= n_col - n_ext - 1:
                axes[i_y, i_x].get_xaxis().set_visible(False)

    else:

        # remove unnecessary axes
        fig.delaxes(axes[i_y, i_x])

    # determine position in grid of axes
    i_x += 1
    if i_x == axes.shape[1]:
        i_x = 0
        i_y += 1

# save plot
fig.subplots_adjust(wspace=0.0, hspace=0.0)
fig.savefig(osp.join(outdir, base_label + '_spin_mass_constraints.pdf'), \
            bbox_inches='tight')

# also generate a distance plot
fig, axes = mp.subplots(n_row, n_col, figsize=(20, height))
g = gdp.get_single_plotter()
i_x = 0
i_y = 0
for i in range(n_row * n_col):
    
    if i < n_targets:

        if skip[i]:
            fig.delaxes(axes[i_y, i_x])
            i_x += 1
            if i_x == axes.shape[1]:
                i_x = 0
                i_y += 1
            continue

        # plot distance and inclination constraints
        g.plot_2d(samples[i], 'distance', 'iota', colors=[cols[i]], \
                  ax=axes[i_y, i_x], filled=True)
        #g.plot_1d(samples[i], 'distance', color=cols[i], \
        #          ax=axes[i_y, i_x])
        label = r'$\rho=' + '{:4.1f}'.format(target_snrs[i]) + '$'
        d_label = r'$d_L^{\rm true}=' + \
                  '{:d}'.format(int(truths[i][4])) + r'\,{\rm Mpc}$'
        if truths[i][3] > np.pi / 2.0:
            text_y = 0.125
            axes[i_y, i_x].text(0.95, 0.125, label, ha='right', va='bottom', \
                                transform=axes[i_y, i_x].transAxes)
            axes[i_y, i_x].text(0.95, 0.04, d_label, ha='right', va='bottom', \
                                transform=axes[i_y, i_x].transAxes)
        else:
            text_y = 0.95
            axes[i_y, i_x].text(0.95, 0.95, label, ha='right', va='top', \
                                transform=axes[i_y, i_x].transAxes)
            axes[i_y, i_x].text(0.95, 0.875, d_label, ha='right', va='top', \
                                transform=axes[i_y, i_x].transAxes)
        axes[i_y, i_x].plot([truths[i][2]], [truths[i][3]], \
                            marker='+', color='k')
        axes[i_y, i_x].grid(False)
        if tight_loc:
            axes[i_y, i_x].set_xlim(-0.125, 0.125)
            axes[i_y, i_x].set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
            axes[i_y, i_x].set_xticklabels(['-0.1', '-0.05', '0', '0.05', '0.1'])
        else:
            axes[i_y, i_x].set_xlim(-0.4, 0.4)
            axes[i_y, i_x].set_xticks([-0.3, -0.15, 0.0, 0.15, 0.3])
            axes[i_y, i_x].set_xticklabels(['-0.3', '-0.15', '0', '0.15', '0.3'])
        axes[i_y, i_x].set_ylim(0.0, np.pi)
        axes[i_y, i_x].set_yticks([0.0, np.pi/4.0, np.pi/2.0, 3.0*np.pi/4.0])
        axes[i_y, i_x].set_yticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$'])

        # remove axis labels where they would otherwise overlap
        if i_x > 0:
            axes[i_y, i_x].get_yaxis().set_visible(False)
        if i_y < n_row - 1:
            if i_y != n_row - 2 or i_x <= n_col - n_ext - 1:
                axes[i_y, i_x].get_xaxis().set_visible(False)

    else:

        # remove unnecessary axes
        fig.delaxes(axes[i_y, i_x])

    # determine position in grid of axes
    i_x += 1
    if i_x == axes.shape[1]:
        i_x = 0
        i_y += 1

# save plot
fig.subplots_adjust(wspace=0.0, hspace=0.0)
fig.savefig(osp.join(outdir, base_label + '_dis_inc_constraints.pdf'), \
            bbox_inches='tight')


# also generate a Lambda plot
fig, axes = mp.subplots(n_row, n_col, figsize=(20, height))
g = gdp.get_single_plotter()
i_x = 0
i_y = 0
for i in range(n_row * n_col):
    
    if i < n_targets:

        if skip[i]:
            fig.delaxes(axes[i_y, i_x])
            i_x += 1
            if i_x == axes.shape[1]:
                i_x = 0
                i_y += 1
            continue

        # plot lambda and mass ratio constraints
        g.plot_2d(samples[i], 'q', 'lambda_2', colors=[cols[i]], \
                  ax=axes[i_y, i_x], filled=True)
        label = r'$\rho=' + '{:4.1f}'.format(target_snrs[i]) + '$'
        axes[i_y, i_x].text(0.95, 0.95, label, ha='right', va='top', \
                            transform=axes[i_y, i_x].transAxes)
        axes[i_y, i_x].plot([truths[i][5]], [truths[i][6]], \
                            marker='+', color='k')
        axes[i_y, i_x].grid(False)
        axes[i_y, i_x].set_xlim(q_inv_min, q_inv_max)
        #axes[i_y, i_x].set_xticks([0.1, 0.2, 0.3])
        if old:
            axes[i_y, i_x].set_ylim(0.0, 4000.0)
        else:
            axes[i_y, i_x].set_ylim(0.0, 4500.0)

        # remove axis labels where they would otherwise overlap
        if i_x > 0:
            axes[i_y, i_x].get_yaxis().set_visible(False)
        if i_y < n_row - 1:
            if i_y != n_row - 2 or i_x <= n_col - n_ext - 1:
                axes[i_y, i_x].get_xaxis().set_visible(False)

    else:

        # remove unnecessary axes
        fig.delaxes(axes[i_y, i_x])

    # determine position in grid of axes
    i_x += 1
    if i_x == axes.shape[1]:
        i_x = 0
        i_y += 1

# save plot
fig.subplots_adjust(wspace=0.0, hspace=0.0)
fig.savefig(osp.join(outdir, base_label + '_lambda_q_constraints.pdf'), \
            bbox_inches='tight')


# also generate a Lambda-tilde plot
fig, axes = mp.subplots(n_row, n_col, figsize=(20, height))
g = gdp.get_single_plotter()
i_x = 0
i_y = 0
for i in range(n_row * n_col):
    
    if i < n_targets:

        if skip[i]:
            fig.delaxes(axes[i_y, i_x])
            i_x += 1
            if i_x == axes.shape[1]:
                i_x = 0
                i_y += 1
            continue

        # plot reduced lambda and spin constraints
        g.plot_2d(samples[i], 'a_1', 'lambda_tilde', colors=[cols[i]], \
                  ax=axes[i_y, i_x], filled=True)
        label = r'$\rho=' + '{:4.1f}'.format(target_snrs[i]) + '$'
        axes[i_y, i_x].text(0.95, 0.95, label, ha='right', va='top', \
                            transform=axes[i_y, i_x].transAxes)
        axes[i_y, i_x].plot([truths[i][1]], [truths[i][7]], \
                            marker='+', color='k')
        axes[i_y, i_x].grid(False)
        axes[i_y, i_x].set_xlim(spin_min_bh, spin_max_bh)
        #axes[i_y, i_x].set_xticks([0.1, 0.3, 0.5, 0.7])
        axes[i_y, i_x].set_ylim(0.0, 200.0)
        axes[i_y, i_x].set_yticks([0.0, 50.0, 100.0, 150.0])

        # remove axis labels where they would otherwise overlap
        if i_x > 0:
            axes[i_y, i_x].get_yaxis().set_visible(False)
        if i_y < n_row - 1:
            if i_y != n_row - 2 or i_x <= n_col - n_ext - 1:
                axes[i_y, i_x].get_xaxis().set_visible(False)

    else:

        # remove unnecessary axes
        fig.delaxes(axes[i_y, i_x])

    # determine position in grid of axes
    i_x += 1
    if i_x == axes.shape[1]:
        i_x = 0
        i_y += 1

# save plot
fig.subplots_adjust(wspace=0.0, hspace=0.0)
fig.savefig(osp.join(outdir, base_label + '_lambda_spin_constraints.pdf'), \
            bbox_inches='tight')
