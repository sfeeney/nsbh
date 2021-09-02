import numpy as np
import bilby
import bilby.gw.conversion as bc
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import matplotlib.patches as mpp
import os
import os.path as osp
import getdist as gd
import getdist.plots as gdp
import lalsimulation as lalsim
import sklearn.decomposition as skd
import ns_eos_aw as nseos
import corner


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

def n_eff_from_log_weights(log_weights):

    # first normalize log-weights
    log_weights = log_weights - np.max(log_weights)
    log_weights_norm = np.log(np.sum(np.exp(log_weights)))
    log_weights = log_weights - log_weights_norm
    weights = np.exp(log_weights)
    n_eff = np.exp(-np.sum(weights * log_weights))
    return n_eff

def lal_inf_sd_gammas_fam(gammas):

    '''
    Modified from LALInferenceSDGammasMasses2Lambdas:
    https://lscsoft.docs.ligo.org/lalsuite/lalinference/_l_a_l_inference_8c_source.html#l02364
    '''

    # create EOS & family
    eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(*gammas)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    
    return fam

def lal_inf_sd_gammas_mass_to_lambda(fam, mass_m_sol):

    '''
    Modified from LALInferenceSDGammasMasses2Lambdas:
    https://lscsoft.docs.ligo.org/lalsuite/lalinference/_l_a_l_inference_8c_source.html#l02364
    '''

    # calculate lambda(m|eos)
    mass_kg = mass_m_sol * m_sol_kg
    rad = lalsim.SimNeutronStarRadius(mass_kg, fam)
    love = lalsim.SimNeutronStarLoveNumberK2(mass_kg, fam)
    comp = big_g * mass_kg / (c ** 2) / rad
    
    return 2.0 / 3.0 * love / comp ** 5

def lal_inf_sd_gammas_mass_to_radius(fam, mass_m_sol):

    '''
    Modified from LALInferenceSDGammasMasses2Lambdas:
    https://lscsoft.docs.ligo.org/lalsuite/lalinference/_l_a_l_inference_8c_source.html#l02364
    '''

    # calculate radius(m|eos) in km
    mass_kg = mass_m_sol * m_sol_kg
    rad = lalsim.SimNeutronStarRadius(mass_kg, fam) / 1.0e3
    
    return rad

def dd2_lambda_from_mass(m):
    return 1.60491e6 - 23020.6 * m**-5 + 194720. * m**-4 - 658596. * m**-3 \
        + 1.33938e6 * m**-2 - 1.78004e6 * m**-1 - 992989. * m + 416080. * m**2 \
        - 112946. * m**3 + 17928.5 * m**4 - 1263.34 * m**5


# plot settings
lw = 1.5
mp.rc('font', family = 'serif')
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# fix a weird LaTeX bug with exponents
mp.rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}'

# constants
m_sol_kg = 1.988409902147041637325262574352366540e30 # LAL_MSUN_SI = LAL_GMSUN_SI / LAL_G_SI
lal_mrsun_si = 1.476625061404649406193430731479084713e3 # LAL_MRSUN_SI = LAL_GMSUN_SI / (LAL_C_SI * LAL_C_SI)
big_g = 6.67430e-11 # m^3/kg/s^2
c = 2.998e8 # m/s
log_zero = -1.0e10
m_ns_std = 1.4 # m_sol

# settings
n_procs = 96
n_eos_samples_per_proc = 12 # 160
n_eos_samples = n_procs * n_eos_samples_per_proc
n_m_samples = 100
n_inds = 4
thin = False
vol_limit = False
mass_prior_volume = False

# data sample settings
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
use_weighted_samples = True
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
outdir = 'outdir'
support_thresh = 1.0e-3

# BH & NS mass prior limits
if uniform_bh_masses:
    m_min_bh = 2.5
    if low_metals:
        m_max_bh = 40.0
    else:
        m_max_bh = 12.0
else:
    m_min_bh = 5.0
    m_max_bh = 20.0
m_min_ns = 1.0
if uniform_ns_masses:
    m_max_ns = 2.42
else:
    m_max_ns = 2.0

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
if vol_limit:
    base_label = base_label + '_vol_lim'
if mass_prior_volume:
    base_label = base_label + '_inc_mpv'


# @TODO: selection! both with and without prior effect
# @TODO: tidy up, inc. module


# useful grid in neutron star mass
n_grid = 1000
m_ns_grid = np.linspace(m_min_ns, m_max_ns, n_grid)

# read in some draws from the gamma prior
gamma_prior_draws = np.genfromtxt('data/ns_eos_gamma_prior_draws.txt', \
                                  delimiter=',')

# read in some mass-lambda curves for EOS prior draws to 
# indicate prior in mass-Lambda space
m_l_prior_draws = np.genfromtxt('data/ns_eos_m_l_prior_draws.txt', \
                                delimiter=',')
m_l_prior_ms = m_l_prior_draws[:, 0]
m_l_prior_ls = m_l_prior_draws[:, 1:]
#l_m_min = np.nanmin(m_l_prior_ls, axis=1)
#l_m_max = np.nanmax(m_l_prior_ls, axis=1)
l_m_min = np.nanpercentile(m_l_prior_ls, 0.5, axis=1)
l_m_max = np.nanpercentile(m_l_prior_ls, 99.5, axis=1)

# read standard-NS lambdas and radii
ns_std_props = np.genfromtxt('data/ns_eos_std_ns_l_r_prior_draws.txt', \
                             delimiter=',')

# read in emcee outputs
filename = osp.join(outdir, base_label + '_eos_emcee_samples.txt')
samples = np.genfromtxt(filename)
if thin:
    with open(filename) as f:
        header = f.readline()
        header = f.readline()
    n_thin = int(header.strip().split(' ')[-1])
else:
    n_thin = 1
n_samples = samples.shape[0]
#pars_in = np.array([0.66613725, 0.4543233, -0.087498, 0.0042616])
#pars_in = np.array([8.089683409048020746e-01, 2.943299329857296254e-01, \
#                    -4.825667425191786097e-02, 1.736697072439619127e-03])
#pars_in = np.array([1.02518133e+00, 1.04993557e-01, \
#                    -1.82736598e-02, 6.00270721e-04])
par_labels = [r'$\gamma_{:d}$'.format(i) for i in range(4)]
limits = np.array([np.min(gamma_prior_draws, axis=0), \
                   np.max(gamma_prior_draws, axis=0)]).T
fig = corner.corner(gamma_prior_draws, plot_datapoints=True, \
                    plot_density=False, plot_contours=False, \
                    labels=par_labels, range=limits, \
                    hist_kwargs={'density': True, 'color': 'grey', \
                                 'ls': '--', 'lw': lw})
corner.corner(samples[::n_thin, 0: 4], plot_datapoints=True, \
                    plot_density=False, plot_contours=False, \
                    labels=par_labels, range=limits, fig=fig, \
                    data_kwargs={'color': 'C0'}, \
                    hist_kwargs={'density': True, 'ec': 'C0', 'lw': lw})
for ax in fig.axes:
    ax.grid(False)
filename = osp.join(outdir, base_label + '_eos_emcee_post.pdf')
fig.savefig(filename, bbox_inches='tight')

# define pca basis
gammas = np.genfromtxt('data/ns_eos_sd_gammas_wysocki.txt', \
                       delimiter=',')
gammas_mean = np.mean(gammas, axis=0)
gammas_std = np.std(gammas, axis=0)
gammas_rs = (gammas - gammas_mean) / gammas_std
pca = skd.PCA(n_components=n_inds)
pca.fit(gammas_rs)
gammas_rs_tf = pca.transform(gammas_rs)
gammas_rs_tf_min = np.min(gammas_rs_tf, axis=0)
gammas_rs_tf_max = np.max(gammas_rs_tf, axis=0)

# loop over (some) samples and plot mass-lambda curves
if not thin:
    n_thin = max(1, int(n_samples / 1000))
n_std = int(np.ceil(n_samples / n_thin))
lambda_ns_std = np.zeros(n_std)
r_ns_std = np.zeros(n_std)
i_std = 0
fig, ax = mp.subplots(1, 1)
for i in range(0, n_samples, n_thin):
    
    # deproject into gamma space
    #thetas = samples[int(n_samples / 2) + i, 0: 4]
    thetas = samples[i, 0: 4]
    gammas = gammas_std * pca.inverse_transform(thetas) + \
             gammas_mean

    # calculate lambdas on mass grid
    n_grid = 1000
    lambdas = np.zeros(n_grid)
    fam = lal_inf_sd_gammas_fam(gammas)
    m_max_eos = lalsim.SimNeutronStarMaximumMass(fam) / m_sol_kg
    masses = np.linspace(m_min_ns, m_max_eos * 0.99999999, n_grid)

    # calculate lambdas on that grid
    for j in range(n_grid):
        lambdas[j] = lal_inf_sd_gammas_mass_to_lambda(fam, masses[j])

    # plot!
    if i == 0:
        mp.plot(masses, lambdas, color='C1', alpha=0.05, \
                label='posterior samples')
    else:
        mp.plot(masses, lambdas, color='C1', alpha=0.05)

    # calculate features of "standard" neutron star
    lambda_ns_std[i_std] = lal_inf_sd_gammas_mass_to_lambda(fam, m_ns_std)
    r_ns_std[i_std] = lal_inf_sd_gammas_mass_to_radius(fam, m_ns_std)
    i_std += 1

# overplot prior and ground truth
mp.fill_between(m_l_prior_ms, l_m_min, 0.0, color='lightgrey', label='non-physical')
#                color='lightgrey', alpha=0.5, zorder=10)
mp.fill_between(m_l_prior_ms, l_m_max, 4500.0, color='lightgrey')
#                color='lightgrey', alpha=0.5, zorder=10)
mp.plot(m_l_prior_ms, dd2_lambda_from_mass(m_l_prior_ms), \
        color='black', ls='--', label='truth')
mp.xlim(m_min_ns, m_max_ns)
mp.ylim(0.0, 4500.0)
mp.xlabel(r'$m_{\rm NS}\,[M_\odot]$')
mp.ylabel(r'$\Lambda_{\rm NS}$')
leg = mp.legend(loc='upper right', handlelength=3.0)
for line in leg.get_lines():
    line.set_linewidth(lw)
    line.set_alpha(1.0)
ax.grid(False)
filename = osp.join(outdir, base_label + '_eos_emcee_mass_lambda_post.pdf')
fig.savefig(filename, bbox_inches='tight')

# calculate radius of DD2 NS with standard mass using AW's code
sim = {'mass1': 10.0, 'spin1x': 0.0, 'spin1y': 0.0, \
       'spin1z': 0.0, 'mass2': m_ns_std, 'spin2x': 0.0, \
       'spin2y': 0.0, 'spin2z': 0.0}
dd2_std_ns = nseos.Foucart(sim, eos="DD2")

# plot constraints on deformability and radius of standard NS
fig, axes = mp.subplots(1, 2, figsize=(10, 5))
axes[0].hist(lambda_ns_std, histtype='step', lw=lw, density=True, \
             label='posterior')
axes[0].hist(ns_std_props[:, 0], histtype='step', lw=lw, \
             color='grey', ls='--', density=True, label='prior')
axes[0].axvline(dd2_lambda_from_mass(m_ns_std), color='C1', ls='-.', \
                label='truth')
axes[0].set_xlabel(r'$\Lambda_{\rm NS}(1.4\,M_\odot)$')
axes[0].set_ylabel(r'${\rm density}$')
leg = axes[0].legend(loc='upper right', handlelength=3.0)
for line in leg.get_lines():
    line.set_linewidth(lw)
axes[1].hist(r_ns_std, histtype='step', lw=lw, density=True)
axes[1].hist(ns_std_props[:, 1], histtype='step', lw=lw, \
             color='grey', ls='--', density=True)
axes[1].axvline(dd2_std_ns.r_ns / 1.0e3, color='C1', ls='-.')
for i in range(2):
    axes[i].grid(False)
    axes[i].yaxis.set_ticks([])
axes[1].set_xlabel(r'$r_{\rm NS}(1.4\,M_\odot)\,[{\rm km}]$')
fig.subplots_adjust(wspace=0, hspace=0)
filename = osp.join(outdir, base_label + '_eos_emcee_l_r_std_post.pdf')
fig.savefig(filename, bbox_inches='tight')

# report percentiles
print('* * *')
print('Lambda(1.4 M_sol) 95% and 68% interval and median:')
print(np.percentile(lambda_ns_std, [2.5, 16.0, 50.0, 84.0, 97.5]))
print('truth:', dd2_lambda_from_mass(m_ns_std))
print('* * *')
print('radius(1.4 M_sol) 95% and 68% interval and median [km]:')
print(np.percentile(r_ns_std, [2.5, 16.0, 50.0, 84.0, 97.5]))
print('truth:', dd2_std_ns.r_ns / 1.0e3)
print('* * *')
