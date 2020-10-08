import numpy as np
import numpy.random as npr
import matplotlib.pyplot as mp
import os.path as osp
import bilby
import pystan as ps
import pickle
import getdist as gd
import getdist.plots as gdp


def d2z(d, h_0, q_0, order=3):

    z = h_0 * d / c
    if order > 1:
        z += -1.0 / 2.0 * (1.0 - q_0) * (h_0 * d / c) ** 2
    if order > 2:
        z += 1.0 / 6.0 * (4.0 - 7.0 * q_0 + 1.0) * (h_0 * d / c) ** 3

    return z

def z2d(z, h_0, q_0, order=3):

    d = c * z / h_0
    if order > 1:
        d += 1.0 / 2.0 * (1.0 - q_0) * c * z ** 2 / h_0
    if order > 2:
        d += 1.0 / 6.0 * (-1.0 + q_0 - 1.0 + 3.0 * q_0 ** 2) * \
             c * z ** 3 / h_0

    return d

def d2z_bisect(z, args):

    d_true, h_0, q_0 = args
    
    return z2d(z, h_0, q_0) - d_true

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

def pom_gmm_bic(model, x_samples):

    # NB np.sum(model.log_probability(x_samples)) can sometimes be NaN
    return 3.0 * model.n * np.log(len(x_samples)) - \
           2.0 * np.sum(model.log_probability(x_samples))

def pom_gmm_aic(model, x_samples):

    # NB np.sum(model.log_probability(x_samples)) can sometimes be NaN
    return 2.0 * 3.0 * model.n - \
           2.0 * np.sum(model.log_probability(x_samples))

def pom_opt_gmm(x_samples, weights, n_comp_min=2, n_comp_max=10, \
                n_rpt=20, thresh=0.0001, use_aic=False):
    
    # loop over number of components
    #high_weights = weights > 0.5 * np.max(weights)
    best_models = []
    best_scores = []
    for n_comp in range(n_comp_min, n_comp_max + 1):

        # repeat fit a few times for each n_comp to find rough optimum
        models = []
        scores = np.zeros(n_rpt)
        for i in range(n_rpt):
            #models.append(pm.gmm.GeneralMixtureModel.from_samples(pm.MultivariateGaussianDistribution, \
            models.append(pm.gmm.GeneralMixtureModel.from_samples(pm.NormalDistribution, \
                                                                  n_components=n_comp, \
                                                                  X=x_samples, \
                                                                  weights=weights, \
                                                                  stop_threshold=thresh))
            if use_aic:
                scores[i] = pom_gmm_aic(models[-1], x_samples)
            else:
                scores[i] = pom_gmm_bic(models[-1], x_samples)
                #scores[i] = pom_gmm_bic(models[-1], x_samples[high_weights, :])
        i_best = np.nanargmin(scores)
        best_models.append(models[i_best])
        best_scores.append(scores[i_best])

    # select and return best overall GMM fit
    i_best = np.argmin(best_scores)
    return best_models[i_best]

# plot settings
lw = 1.5
mp.rc('font', family='serif', size=10)
mp.rcParams['text.latex.preamble'] = [r'\boldmath']
mp.rcParams['axes.linewidth'] = lw
mp.rcParams['lines.linewidth'] = lw

# settings
c = 2.998e5 # km / s
h_0 = 67.36 # km / s / Mpc
q_0 = 0.5 * 0.3153 - 0.6847
sig_v_pec = 500.0 # LVC take U(-1000,1000)
sig_v_pec_obs = 200.0 # km/s
sig_z_obs = 0.001 # 300 km/s
snr_thresh = 12.0
d_max = 2045.0 # Mpc
z_max = d2z(d_max, h_0, q_0)
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
if use_polychord:
    n_live = 1000
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
seobnr_waveform = True
if seobnr_waveform:
    waveform_approximant = 'SEOBNRv4_ROM_NRTidalv2_NSBH'
    aligned_spins = True
else:
    waveform_approximant = 'IMRPhenomPv2_NRTidal'
    aligned_spins = False
lam_det_test = False
datdir = 'data'
outdir = 'outdir'
fit_d_dists = False
kde_fit = False
bw_grid = np.logspace(-0.5, 3.5, 10)
#bw_grid = np.logspace(-0.5, 3.5, 100)
test_stan = False
recompile = False
constrain = True

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
par_file = base_label + '.txt'
raw_pars = np.genfromtxt(datdir + '/' + par_file, \
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
target_redshifts = raw_pars['redshift'][i_sort]
n_targets = len(target_ids)

# optionally test Stan sampling of distance distributions
if test_stan:
    if kde_fit:
        stub = 'kde'
    else:
        stub = 'gmm'
    if recompile:
        stan_model = ps.StanModel(stub + '_sampling.stan')
        with open(stub + '_sampling.pkl', 'wb') as f:
            pickle.dump(stan_model, f)
    else:
        try:
            with open(stub + '_sampling.pkl', 'rb') as f:
                stan_model = pickle.load(f)
        except EnvironmentError:
            print('ERROR: pickled Stan model ' + \
                  '(' + stub + '_sampling.pkl) ' + \
                  'not found. Please set recompile = True')
            exit()

# optionally read in fitted KDE/GMM parameters
n_comp = np.zeros(n_targets, dtype=int)
d_comp_weights = []
d_comp_means = []
if not kde_fit:
    d_comp_stds = []
if not fit_d_dists:
    if use_polychord:
        fit_file = 'pc_'
    else:
        fit_file = ''
    if kde_fit:
        fit_file = outdir + '/' + fit_file + base_label + '_kde_fits.txt'
        d_bws = np.genfromtxt(fit_file, dtype=None, names=True, \
                              delimiter=',', encoding=None)['bandwidth']
    else:

        # read in raw fit data
        fit_file = outdir + '/' + fit_file + base_label + '_gmm_fits.txt'
        fit_data = np.genfromtxt(fit_file, dtype=None, names=True, \
                                 delimiter=',', encoding=None)
        
        # and reform into weights, means, and std devs (bit tortuous)
        n_lines = len(fit_data['id'])
        inds = np.where(np.roll(fit_data['id'],1)!=fit_data['id'])[0]
        n_targets = len(inds)
        inds = np.append(inds, n_lines)
        for i in range(n_targets):
            n_comp[i] = inds[i + 1] - inds[i]
            d_comp_weights.append(fit_data['weight'][inds[i]:inds[i + 1]])
            d_comp_means.append(fit_data['mean'][inds[i]:inds[i + 1]])
            d_comp_stds.append(fit_data['stddev'][inds[i]:inds[i + 1]])

else:

    # otherwise perform required imports
    if kde_fit:
        import sklearn.neighbors as skn
        import sklearn.model_selection as skms
    else:
        import pomegranate as pm

# optionally fix random seed
if constrain:
    npr.seed(161222)

# set up per-object plots
n_col = 8
n_row = int(np.ceil(n_targets / float(n_col)))
height = 2.86 * n_row
n_ext = n_col * n_row - n_targets
fig, axes = mp.subplots(n_row, n_col, figsize=(20, height))

# loop over targets
#n_targets = 5
truths = []
skip = np.full(n_targets, False)
if kde_fit:
    if fit_d_dists:
        d_bws = np.zeros(n_targets)
    else:
        d_bws = d_bws[0: n_targets]
d_l = np.zeros(n_targets)
z_cos = np.zeros(n_targets)
v_pec = np.zeros(n_targets)
v_pec_obs = np.zeros(n_targets)
z_obs = np.zeros(n_targets)
# @TODO REMOVE
##run_ids = [34, 41, 61, 67, 97]
#run_ids = [41, 61, 97]
#for i in run_ids:
# @TODO REMOVE
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
    #label = label_str.format(target_ids[i], duration, minimum_frequency, \
    #                         reference_frequency)
    res_file = label + '_result.json'
    print(osp.join(outdir, res_file))
    if not osp.exists(osp.join(outdir, res_file)):
        skip[i] = True
        truths.append(None)
        n_comp[i] = 1
        d_comp_weights.append(np.zeros(1))
        d_comp_means.append(np.zeros(1))
        if not kde_fit:
            d_comp_stds.append(np.zeros(1))
        continue
    result = bilby.result.read_in_result(filename=osp.join(outdir, res_file))

    # true distance and redshift
    d_l[i] = result.injection_parameters['luminosity_distance']
    z_cos[i] = target_redshifts[i]

    # draw true peculiar velocity and calculate total redshift
    v_pec[i] = npr.randn() * sig_v_pec
    z_tot = z_cos[i] + (1.0 + z_cos[i]) * v_pec[i] / c

    # simulate noisy observations
    v_pec_obs[i] = v_pec[i] + npr.randn() * sig_v_pec_obs
    z_obs[i] = z_tot + npr.randn() * sig_z_obs

    # extract posterior samples relevant for reweighting
    d_l_samples = result.posterior.luminosity_distance
    m_c_samples = result.posterior.chirp_mass
    q_inv_samples = result.posterior.mass_ratio
    m_1_samples, m_2_samples = \
        chirp_q_to_comp_masses(m_c_samples, q_inv_samples)

    # define importance weights. we want to convert from
    # 1) the prior we used, which is uniform in chirp mass and mass 
    # ratio, and is a uniform source frame prior in distance
    # to
    # 2) the prior we want for the hierarchical analysis, which is 
    # uniform in component masses and distance
    d_l_prior = \
        bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', \
                                          minimum=10.0, maximum=2500.0, \
                                          unit='Mpc', boundary=None)
    d_l_weights = 1.0 / d_l_prior.prob(d_l_samples)
    mass_weights = prior_change_jac(m_c_samples, q_inv_samples)
    m_1_mask = np.logical_and(m_1_samples >= m_min_bh, \
                              m_1_samples <= m_max_bh)
    m_2_mask = np.logical_and(m_2_samples >= m_min_ns, \
                              m_2_samples <= m_max_ns)
    m12_mask = np.logical_and(m_1_mask, m_2_mask)
    mass_weights *= m12_mask
    mass_weights += ~m12_mask * 1.0e-10
    weights = d_l_weights * mass_weights
    weights /= np.sum(weights)

    # KDE/GMM fit
    d_l_grid = np.linspace(0.95 * np.min(d_l_samples), \
                           1.05 * np.max(d_l_samples), 1000)
    if kde_fit:
        n_comp[i] = len(d_l_samples)
        d_comp_means.append(d_l_samples)
        d_comp_weights.append(weights)
    if fit_d_dists:

        if kde_fit:

            # KDE = one component per sample, with the same weight and 
            # position. need gridsearch to find optimal bandwidth
            gs = skms.GridSearchCV(skn.KernelDensity(), \
                                   {'bandwidth': bw_grid}, \
                                   cv=20)
            gs.fit(d_l_samples[:, None], sample_weight=weights)
            kde = gs.best_estimator_
            pdf = np.exp(kde.score_samples(d_l_grid[:, None]))
            d_bws[i] = kde.bandwidth

        else:

            # lower-dimensionality (2-10-component) GMM fit
            gmm = pom_opt_gmm(d_l_samples[:, None], weights, use_aic=True)
            pdf = np.exp(gmm.log_probability(d_l_grid))

            # extract fit parameters: optimum number of components, and 
            # their weights, means and standard deviations
            n_comp[i] = gmm.n
            d_comp_weights.append(np.exp(gmm.weights))
            d_comp_means.append(np.zeros(gmm.n))
            d_comp_stds.append(np.zeros(gmm.n))
            for n in range(gmm.n):
                d_comp_means[-1][n] = gmm.distributions[n].parameters[0]
                d_comp_stds[-1][n] = gmm.distributions[n].parameters[1]

    else:

        # put together pdf from previous results
        pdf = np.zeros(1000)
        if kde_fit:
            
            for j in range(n_comp[i]):
                pdf += np.exp(-0.5 * ((d_l_grid - d_l_samples[j]) / \
                                      d_bws[i]) ** 2) / \
                       np.sqrt(2.0 * np.pi) / d_bws[i] * weights[j]
            pdf /= np.sum(weights)

        else:

            for j in range(n_comp[i]):
                pdf += np.exp(-0.5 * ((d_l_grid - d_comp_means[i][j]) / \
                                      d_comp_stds[i][j]) ** 2) / \
                       np.sqrt(2.0 * np.pi) / d_comp_stds[i][j] * \
                       d_comp_weights[i][j]

    # plot indices
    i_x = i % n_col
    i_y = i // n_col

    # plot effects of reweighting, using simple histograms for now
    axes[i_y, i_x].plot(d_l_grid, pdf, color='k')
    hist, bin_edges = np.histogram(d_l_samples, bins=25, \
                                   density=True)
    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    axes[i_y, i_x].plot(bin_centres, hist, color='C0')
    whist, bin_edges = np.histogram(d_l_samples, bins=25, \
                                   density=True, weights=mass_weights)
    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    axes[i_y, i_x].plot(bin_centres, whist, color='C1', ls='-.')
    whist, bin_edges = np.histogram(d_l_samples, bins=25, \
                                   density=True, weights=weights)
    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    axes[i_y, i_x].plot(bin_centres, whist, color='C2', ls='--')
    axes[i_y, i_x].grid(False)

    # optionally sample from distance distributions using stan
    if test_stan:
        if kde_fit:
            stan_data = {'n_comp': len(d_l_samples), \
                         'locs': d_l_samples, \
                         'weights': weights, \
                         'bw': kde.bandwidth}
        else:
            stan_data = {'n_comp': n_comp[i], \
                         'weights': d_comp_weights[-1], \
                         'locs': d_comp_means[-1], \
                         'stds': d_comp_stds[-1]}
        fit = stan_model.sampling(data=stan_data, iter=10000, \
                                  control={'adapt_delta':0.9})
        stan_samples = fit.extract(permuted=True, inc_warmup=False)
        shist, bin_edges = np.histogram(stan_samples['d'], bins=25, \
                                        density=True)
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        axes[i_y, i_x].plot(bin_centres, shist, color='C4', ls=':')

    # labeling is a bit complicated...
    axes[i_y, i_x].set_yticks([])
    if i_x == 0:
        axes[i_y, i_x].set_ylabel(r'${\rm P}(d_L|\hat{h})$')
    if i_y == n_row - 2 and i_x > n_col - n_ext - 1:
        axes[i_y, i_x].set_xlabel(r'$d_L\,[{\rm Mpc}]$')
    elif i_y == n_row - 1:
        axes[i_y, i_x].set_xlabel(r'$d_L\,[{\rm Mpc}]$')

    # findings
    # 1 - precise normalization of weights doesn't matter for KDE! 
    #     weights = ones or 2 * ones is exactly the same
    # 2 - weighted histogram looks like weighted KDE.
    # 3 - impact of weights on histogram makes sense
    # 4 - some of these posteriors have large numbers of zero-weight
    #     samples. what to do? try to sample properly? plot all posts
    #     with and without weighting?
    # 5 - sample weights apply directly to KDE

# finish posterior fit plot
if use_polychord:
    base_label = 'pc_' + base_label
for i in range(n_targets, n_row * n_col):
    i_x = i % n_col
    i_y = i // n_col
    fig.delaxes(axes[i_y, i_x])
plot_file = outdir + '/' + base_label + '_d_l_post_'
if kde_fit:
    plot_file = plot_file + 'kde_fits'
else:
    plot_file = plot_file + 'gmm_fits'
if test_stan:
    plot_file = plot_file + '_stan_test'
fig.savefig(plot_file + '.pdf', bbox_inches='tight')

# optionally save KDE bandwidth fits
if fit_d_dists:

    if kde_fit:
        
        fmt = '{:d},{:.9e}'
        with open(outdir + '/' + base_label + '_kde_fits.txt', 'w') as f:
            f.write('#id,bandwidth')
            for i in range(n_targets):
                f.write('\n' + fmt.format(target_ids[i], d_bws[i]))

    else:
        
        fmt = '{:d},{:.9e},{:.9e},{:.9e}'
        with open(outdir + '/' + base_label + '_gmm_fits.txt', 'w') as f:
            f.write('#id,weight,mean,stddev')
            for i in range(n_targets):
                for n in range(n_comp[i]):
                    f.write('\n' + fmt.format(target_ids[i], \
                                              d_comp_weights[i][n], \
                                              d_comp_means[i][n], \
                                              d_comp_stds[i][n]))

# @TODO: fudged n_bar
n_bar_det_coeffs = np.array([n_targets] + [0] * 14)

# bit of book-keeping of KDE outputs
n_comp_max = np.max(n_comp)
d_weights = np.zeros((n_comp_max, n_targets))
d_means = np.zeros((n_comp_max, n_targets))
d_stds = np.zeros((n_comp_max, n_targets))
for i in range(n_targets):
    d_weights[0: n_comp[i], i] = d_comp_weights[i]
    d_means[0: n_comp[i], i] = d_comp_means[i]
    if kde_fit:
        d_stds[0: n_comp[i], i] = d_bws[i] * np.ones(n_comp[i])
    else:
        d_stds[0: n_comp[i], i] = d_comp_stds[i]
stan_data = {'fixed_n_mrg': 1, 'n_mrg': n_targets, \
             'n_cmp_max': n_comp_max, 'n_cmp': n_comp, \
             'obs_d_weights': d_weights, 'obs_d_means': d_means, \
             'obs_d_stds': d_stds, 'obs_v_pec': v_pec_obs, \
             'obs_z': z_obs, 'sig_v_pec': sig_v_pec, \
             'sig_obs_v_pec': sig_v_pec_obs, 'sig_z': sig_z_obs, \
             'z_max': z_max, 'n_coeffs': 15, \
             'n_bar_det_coeffs': n_bar_det_coeffs}

# sample
if recompile:
    stan_model = ps.StanModel('nsbh_cosmo.stan')
    with open('nsbh_cosmo.pkl', 'wb') as f:
        pickle.dump(stan_model, f)
else:
    try:
        with open('nsbh_cosmo.pkl', 'rb') as f:
            stan_model = pickle.load(f)
    except EnvironmentError:
        print('ERROR: pickled Stan model ' + \
              '(nsbh_cosmo.pkl) ' + \
              'not found. Please set recompile = True')
        exit()
#fit = stan_model.sampling(data=stan_data, iter=1000)
fit = stan_model.sampling(data=stan_data, iter=10000)
#fit = stan_model.sampling(data=stan_data, iter=10000, \
#                          control={'adapt_delta':0.9})
print(fit)
raw_samples = fit.extract(permuted=False, inc_warmup=False)
n_pars = raw_samples.shape[2] - 1
n_chains = raw_samples.shape[1]
n_samples = raw_samples.shape[0]
samples = np.zeros((n_chains * n_samples, n_pars))
for i in range(0, n_chains):
    for j in range(0, n_pars):
        samples[i * n_samples: (i + 1) * n_samples, j] = raw_samples[:, i, j]

# @TODO: save samples

# cheeky plot
pars = ['h_0', 'q_0']
par_names = ['H_0', 'q_0']
par_vals = [h_0, q_0]
gd_samples = gd.MCSamples(samples=samples[:, 0: 2], names=pars, 
                          labels=par_names, ranges={})
g = gdp.getSubplotPlotter()
g.settings.lw_contour = lw
g.settings.axes_fontsize = 8
g.triangle_plot(gd_samples, pars, filled = True, \
                line_args = {'lw': lw, 'color': 'C0'}, \
                contour_args = {'lws': [lw, lw]}, \
                colors = ['C0'])
for i in range(0, len(pars)):
    sp_title = '$' + gd_samples.getInlineLatex(pars[i], \
                                               limit=1) + '$'
    g.subplots[i, i].set_title(sp_title, fontsize=12)
    for ax in g.subplots[i, :i]:
        ax.axhline(par_vals[i], color='gray', ls='--')
        ax.grid(False)
    for ax in g.subplots[i:, i]:
        ax.axvline(par_vals[i], color='gray', ls='--')
        ax.grid(False)
plot_file = outdir + '/' + base_label
if kde_fit:
    plot_file = plot_file + '_kde_fits'
else:
    plot_file = plot_file + '_gmm_fits'
mp.savefig(plot_file + '_cosmo_post_triangle_plot.pdf', bbox_inches='tight')


exit()


# @TODO
# 1) DONE: tidy plot name
# 2) DONE: add true and observed peculiar velocities
# 3) DONE: recall true redshift and add noise
# 4) DONE: save true redshifts... will have to rerun everything FFS. just back out for now?
# 5) fix up n_bar
# 6) try sampling for five or ten mergers
# 7) rate sampling!
