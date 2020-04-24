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

# update plot coordinates
def update_xy(i_x, i_y, n_x=5):
    if i_x == n_x - 1:
        i_x = 0
        i_y += 1
    else:
        i_x +=1
    return (i_x, i_y)

# plot an individual 1D Lambda posterior
def plot_1d(gd_plotter, ax, samples, truth, label, flag=False, \
            label_y=False):
    if flag:
        col = 'k'
    else:
        col = 'C0'
    gd_plotter.plot_1d(samples, 'lambda_2', ax=ax, colors=[col])
    ax.axvline(truth, color='C1')
    ax.set_xlim(0.0, 4000.0)
    ax.set_xticks([0.0, 1000.0, 2000.0, 3000.0])
    if label_y:
        ax.get_yaxis().set_visible(True)
        ax.set_ylabel(r'${\rm P}(\Lambda_{\rm NS})$')
    ax.grid(False)
    ax.text(0.99, 0.98, label, transform=ax.transAxes, \
            fontsize=12, va='top', ha='right')#, \
    #        bbox={'fc': 'white', 'ec':'black'})

# plot a batch of 1D Lambda posterior variants
def plot_1ds(n_plot, i_plot, i_x, i_y, i_xy_base, gd_plotter, \
             fig, axes, skip, samples, truths, labels):

    # loop over all samples to plot, checking whether we have a 
    # posterior to plot (i.e., if skip set to False)
    for i in range(n_plot):

        # insert base model constraints at appropriate points
        if [i_y, i_x] in i_xy_base:
            if skip[0]:
                fig.delaxes(axes[i_y, i_x])
            else:
                plot_1d(gd_plotter, axes[i_y, i_x], samples[0], \
                        truths[0][0], labels[0], flag=True, \
                        label_y=not bool(i_x))
                if i_y > 0:
                    try:
                        axes[i_y - 1, i_x].get_xaxis().set_visible(False)
                    except:
                        pass
            i_x, i_y = update_xy(i_x, i_y)

        # plot variant run constraints
        if skip[i_plot]:
            fig.delaxes(axes[i_y, i_x])
        else:
            plot_1d(gd_plotter, axes[i_y, i_x], samples[i_plot], \
                    truths[i_plot][0], labels[i_plot], label_y=not bool(i_x))
            if i_y > 0:
                try:
                    axes[i_y - 1, i_x].get_xaxis().set_visible(False)
                except:
                    pass
        i_x, i_y = update_xy(i_x, i_y)
        i_plot += 1

    # remove all unecessary axes
    if i_x > 0:
        for i_x in range(i_x, 5):
            fig.delaxes(axes[i_y, i_x])

    return i_plot

# pick out a set of axes
def flag_axes(axes, i_xy):
    for (i_y, i_x) in i_xy:
        axes[i_y, i_x].tick_params(color='black')
        for spine in axes[i_y, i_x].spines.values():
            spine.set_edgecolor('black')


# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# common settings
datdir = 'data'
outdir = 'outdir'
base_label = 'lam_det_test'
zero_spins = False
tight_loc = False
fixed_ang = True
n_live = 1000

# read list of all targets
targets = np.genfromtxt(osp.join(datdir, base_label + '.txt'), \
                        dtype=None, names=True, delimiter=',', \
                        encoding=None)
n_targets = len(targets)

# loop over targets
samples = []
truths = []
skip = np.full(n_targets, False)
for i in range(n_targets):
#for i in range(9):

    # read in results file, which contains tonnes of info
    label = base_label + '_inj_{:d}'.format(i)
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
        print('run', i, 'incomplete')
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
    truths.append([all_pars['lambda_2'], \
                   all_pars['a_1'], \
                   all_pars['iota'], \
                   all_pars['mass_ratio'], \
                   all_pars['chirp_mass']])
    
    # convert to GetDist MCSamples object
    try:
        test = result.posterior
    except ValueError:
        skip[i] = True
        samples.append(None)
        print('run', i, 'incomplete')
        continue
    gd_samples = np.array([result.posterior.lambda_2, \
                           result.posterior.a_1, \
                           result.posterior.iota, \
                           result.posterior.mass_ratio, \
                           result.posterior.chirp_mass]).T
    samples.append(gd.MCSamples(samples=gd_samples, \
                                names=['lambda_2', 'a_1', 'iota', \
                                       'q', 'm_c'], \
                                labels=[r'\Lambda_{\rm NS}', 'a_1', \
                                        r'\iota', 'm_2/m_1', \
                                        r'\mathcal{M}_{\rm c}'], \
                                ranges={'a_1':(0.0, 0.8), \
                                        'iota':(0.0, np.pi), \
                                        'q':(0.02, 0.6), \
                                        'lambda_2':(0.0, 4000.0)}))

# snr-ordered colouring
n_targets = len(samples)
cm = mpcm.get_cmap('plasma')
cols = [mpc.rgb2hex(cm(x)) for x in np.linspace(0.2, 0.8, n_targets)[::-1]]

# generate figure and axes
n_col = 5
n_row = 8
fig, axes = mp.subplots(n_row, n_col, figsize=(20, 30))
g = gdp.get_single_plotter()

# set up points at which we want to plot the base run
i_xy_base = [[0, 1], [1, 0], [2, 1], [4, 4], [6, 4]]

# nummbers of different variants
n_spin_mag = 4
n_spin_dir = 2
n_inc = 5
n_q = 6

# @TODO
# 3 - credible intervals?

# plot spin magnitude variants
spin_mags = np.sqrt(targets['spin1x'] ** 2 + \
                    targets['spin1y'] ** 2 + \
                    targets['spin1z'] ** 2)
labels = ['$|a|={:4.2f}$'.format(sm) for sm in spin_mags]
i_plot = plot_1ds(n_spin_mag, 1, 0, 0, i_xy_base, g, fig, axes, \
                  skip, samples, truths, labels)

# plot spin direction variants
labels = [''] * n_targets
labels[0] = r'$\hat{\mathbf{a}} \simeq \hat{\mathbf{z}}$'
labels[i_plot] = r'$\hat{\mathbf{a}} \simeq \hat{\mathbf{x}}$'
labels[i_plot + 1] = r'$\hat{\mathbf{a}} \simeq \hat{\mathbf{y}}$'
i_plot = plot_1ds(n_spin_dir, i_plot, 0, 1, i_xy_base, g, fig, axes, \
                  skip, samples, truths, labels)

# plot inclination variants
labels = [r'$\iota' + '={:4.2f}$'.format(inc) \
          for inc in targets['inclination']]
i_plot = plot_1ds(n_inc, i_plot, 0, 2, i_xy_base, g, fig, axes, \
                  skip, samples, truths, labels)

# plot mass ratio variants (with fixed distances)
labels = ['$q={:4.2f}$ (fixed $d$)'.format(q) \
          for q in targets['mass1'] / targets['mass2']]
i_plot = plot_1ds(n_q, i_plot, 0, 4, i_xy_base, g, fig, axes, \
                  skip, samples, truths, labels)

# plot mass ratio variants (with fixed SNRs)
labels = ['$q={:4.2f}$ (fixed '.format(q) + r'$\rho$)' \
          for q in targets['mass1'] / targets['mass2']]
i_plot = plot_1ds(n_q, i_plot, 0, 6, i_xy_base, g, fig, axes, \
                  skip, samples, truths, labels)

# bit of highlighting
#flag_axes(axes, i_xy_base)

# save plot
fig.subplots_adjust(wspace=0.0, hspace=0.0)
fig.savefig(osp.join(outdir, base_label + '_constraints.pdf'), \
            bbox_inches='tight')

