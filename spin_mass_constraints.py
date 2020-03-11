import numpy as np
import bilby
import bilby.gw.conversion as bc
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import os.path as osp
import getdist as gd
import getdist.plots as gdp
import copy
#import matplotlib
#matplotlib.use('TkAgg')

# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# common settings
duration = 32.0
sampling_frequency = 2048.
minimum_frequency = 20.0
reference_frequency = 14.0
ifo_list = ['H1', 'L1', 'V1', 'K1'] # ['H1', 'L1', 'V1']
zero_spins = False
remnants_only = True
tight_loc = True
outdir = 'outdir'
if ifo_list == ['H1', 'L1', 'V1']:
    ifo_str = ''
else:
    ifo_str = '_'.join(ifo_list) + '_'
label_str = 'nsbh_inj_' + ifo_str + \
            '{:d}_d_{:04.1f}_mf_{:4.1f}_rf_{:4.1f}'
if zero_spins:
    label_str += '_zero_spins'
if tight_loc:
    label_str += '_tight_loc'

# read list of all targets
targets = np.genfromtxt('data/remnant_sorted_detected.txt', delimiter=' ')
target_ids = targets[:, 0].astype(int)
target_snrs = targets[:, 2]
if remnants_only:
    target_ids = target_ids[targets[:, 1] > 0.0]
    target_snrs = target_snrs[targets[:, 1] > 0.0]

# loop over targets
n_targets = len(target_ids)
samples = []
truths = []
skip = np.full(n_targets, False)
for i in range(n_targets):
#for i in range(9):

    # read in results file, which contains tonnes of info
    label = label_str.format(target_ids[i], duration, minimum_frequency, \
                             reference_frequency)
    res_file = label + '_result.json'
    print(osp.join(outdir, res_file))
    if not osp.exists(osp.join(outdir, res_file)):
        skip[i] = True
        samples.append(None)
        truths.append(None)
        continue
    result = bilby.result.read_in_result(filename=osp.join(outdir, res_file))
    '''truths.append([result.injection_parameters['mass_1'], \
                   result.injection_parameters['a_1'], 0.0, \
                   result.injection_parameters['iota'], \
                   result.injection_parameters['luminosity_distance'], \
                   result.injection_parameters['mass_ratio'], \
                   result.injection_parameters['lambda_2'], \
                   result.injection_parameters['lambda_tilde']])'''
    all_pars = bc.generate_all_bns_parameters(result.injection_parameters)
    truths.append([all_pars['mass_1'], \
                   all_pars['a_1'], 0.0, \
                   all_pars['iota'], \
                   all_pars['luminosity_distance'], \
                   all_pars['mass_ratio'], \
                   all_pars['lambda_2'], \
                   all_pars['lambda_tilde']])
    
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
                                       'q', 'lambda_2', 'lambda_tilde'], \
                                labels=['a_1', 'm_1', \
                                        distance_label, r'\iota', \
                                        'm_2/m_1', r'\Lambda_{\rm NS}', \
                                        r'\tilde{\Lambda}'], \
                                ranges={'a_1':(0.0, 0.8), \
                                        'iota':(0.0, np.pi), \
                                        'q':(0.02, 0.4), \
                                        'lambda_2':(0.0, 4000.0), \
                                        'lambda_tilde':(0.0, None)}))

# snr-ordered colouring
n_targets = len(samples)
cm = mpcm.get_cmap('plasma')
cols = [mpc.rgb2hex(cm(x)) for x in np.linspace(0.2, 0.8, n_targets)[::-1]]

# generate figure and plot!
n_col = 8
n_row = int(np.ceil(n_targets / float(n_col)))
n_ext = n_col * n_row - n_targets
fig, axes = mp.subplots(n_row, n_col, figsize=(20, 20))
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
        g.plot_2d(samples[i], 'mass_1', 'a_1', colors=[cols[i]], \
                  ax=axes[i_y, i_x], filled=True)
        label = r'$\rho=' + '{:4.1f}'.format(target_snrs[i]) + '$'
        axes[i_y, i_x].plot([truths[i][0]], [truths[i][1]], \
                            marker='+', color='k')
        axes[i_y, i_x].text(7.75, 0.725, label)
        axes[i_y, i_x].grid(False)
        axes[i_y, i_x].set_xlim(4.0, 10.0)
        axes[i_y, i_x].set_ylim(0.0, 0.8)
        axes[i_y, i_x].set_xticks([5.0, 7.0, 9.0])
        axes[i_y, i_x].set_yticks([0.1, 0.3, 0.5, 0.7])

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
fig.savefig(osp.join(outdir, 'spin_mass_constraints.pdf'), bbox_inches='tight')


# also generate a distance plot
fig, axes = mp.subplots(n_row, n_col, figsize=(20, 20))
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
        g.plot_2d(samples[i], 'distance', 'iota', colors=[cols[i]], \
                  ax=axes[i_y, i_x], filled=True)
        #g.plot_1d(samples[i], 'distance', color=cols[i], \
        #          ax=axes[i_y, i_x])
        label = r'$\rho=' + '{:4.1f}'.format(target_snrs[i]) + '$'
        d_label = r'$d_L^{\rm true}=' + \
                  '{:d}'.format(int(truths[i][4])) + r'\,{\rm Mpc}$'
        if truths[i][3] > 2.2:
            axes[i_y, i_x].text(0.12, 0.5, label, ha='right')
            axes[i_y, i_x].text(0.12, 0.2, d_label, ha='right')
        else:
            axes[i_y, i_x].text(0.12, 2.8, label, ha='right')
            axes[i_y, i_x].text(0.12, 2.5, d_label, ha='right')
        axes[i_y, i_x].plot([truths[i][2]], [truths[i][3]], \
                            marker='+', color='k')
        axes[i_y, i_x].grid(False)
        axes[i_y, i_x].set_xlim(-0.125, 0.125)
        axes[i_y, i_x].set_ylim(0.0, np.pi)
        axes[i_y, i_x].set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
        axes[i_y, i_x].set_xticklabels(['-0.1', '-0.05', '0', '0.05', '0.1'])
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
fig.savefig(osp.join(outdir, 'dis_inc_constraints.pdf'), bbox_inches='tight')


# also generate a Lambda plot
fig, axes = mp.subplots(n_row, n_col, figsize=(20, 20))
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
        g.plot_2d(samples[i], 'q', 'lambda_2', colors=[cols[i]], \
                  ax=axes[i_y, i_x], filled=True)
        label = r'$\rho=' + '{:4.1f}'.format(target_snrs[i]) + '$'
        axes[i_y, i_x].text(0.26, 3700, label)
        axes[i_y, i_x].plot([truths[i][5]], [truths[i][6]], \
                            marker='+', color='k')
        axes[i_y, i_x].grid(False)
        axes[i_y, i_x].set_xlim(0.02, 0.4)
        axes[i_y, i_x].set_xticks([0.1, 0.2, 0.3])
        axes[i_y, i_x].set_ylim(0.0, 4000.0)

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
fig.savefig(osp.join(outdir, 'lambda_q_constraints.pdf'), bbox_inches='tight')


# also generate a Lambda-tilde plot
fig, axes = mp.subplots(n_row, n_col, figsize=(20, 20))
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
        g.plot_2d(samples[i], 'a_1', 'lambda_tilde', colors=[cols[i]], \
                  ax=axes[i_y, i_x], filled=True)
        label = r'$\rho=' + '{:4.1f}'.format(target_snrs[i]) + '$'
        axes[i_y, i_x].text(0.5, 180, label)
        axes[i_y, i_x].plot([truths[i][1]], [truths[i][7]], \
                            marker='+', color='k')
        axes[i_y, i_x].grid(False)
        axes[i_y, i_x].set_xlim(0.0, 0.8)
        axes[i_y, i_x].set_xticks([0.1, 0.3, 0.5, 0.7])
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
fig.savefig(osp.join(outdir, 'lambda_spin_constraints.pdf'), bbox_inches='tight')
