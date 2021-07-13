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

# settings
n_procs = 96
n_eos_samples_per_proc = 12 # 160
n_eos_samples = n_procs * n_eos_samples_per_proc
n_m_samples = 100
n_inds = 4
log_zero = -1.0e10

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
seobnr_waveform = True
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

# useful grid in neutron star mass
m_ns_grid = np.linspace(m_min_ns, m_max_ns, 1000)

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





# very temporary
# read in emcee outputs
import corner
samples = np.genfromtxt('temp.txt')
print(samples.shape)
pars_in = np.array([0.66613725, 0.4543233, -0.087498, 0.0042616])
pars_in = np.array([8.089683409048020746e-01, 2.943299329857296254e-01, -4.825667425191786097e-02, 1.736697072439619127e-03])
pars_in = np.array([1.02518133e+00, 1.04993557e-01, -1.82736598e-02, 6.00270721e-04])
par_labels = [r'$\gamma_{:d}$'.format(i) for i in range(4)]
fig = corner.corner(samples[:, 0: 4], plot_datapoints=True, \
                    plot_density=False, plot_contours=False, \
                    truths=pars_in, labels=par_labels)
for ax in fig.axes:
    ax.grid(False)
fig.savefig('temp_post.pdf')

exit()





# @TODO
# DONE: how many mass / EOS samples have non-log-zero likelihoods
# how many sources am i using in comparison to Landry et al.?
# DONE: plot of weights
# DONE: n_eff per eos
# 2D plot of Lambda and m_ns indicating sampling range
# smaller subset of best m/lambda constraints (but Lambda is derived...)
# DONE: check out bilby EOS sampling - seems to be single event only
# what's actually the issue here? that it's hard to get an EOS that 
#  actually matches all of the data well? that the data are therefore 
#  constraining? that the exploration is therefore super inefficient?



# read in datafile. do we need to compile multiple files?
n_runs = 0
compile = True
if compile:

    # read in files
    data_list = []
    for i in range(n_procs):
        stub = '_eos_samples_{:d}_of_{:d}.txt'.format(i, n_procs)
        data_file = osp.join(outdir, base_label + stub)
        data = np.genfromtxt(data_file)
        data_list.append(data)
        n_runs += data.shape[0]
    data = np.concatenate(data_list)
    n_targets = data.shape[1] - n_inds - 2

    # store processed data
    np.savetxt(osp.join(outdir, base_label + '_eos_samples.txt'), data)
    fname = osp.join(outdir, base_label + \
                     '_eos_samples_*_of_{:d}.txt'.format(n_procs))
    print('results compiled. consider "rm ' + fname + '"')

else:

    # read in compiled results
    data_file = osp.join(outdir, base_label + '_eos_samples.txt')
    data = np.genfromtxt(data_file)
    n_runs = data.shape[0]
    n_targets = data.shape[1] - n_inds - 2

# how many samples have non-log-zero likelihoods? seems like roughly half
non_log_zero = data[:, 0] > (log_zero / 10.0)
print(np.sum(non_log_zero), '/', len(non_log_zero), 'non-log-zero samples')
#mp.hist(np.exp(all_samples[non_log_zero, 0]), bins=50)
#mp.show()

# n_eff for full sample and by EOS
print(n_eff_from_log_weights(data[:, 0]))
print(n_eff_from_log_weights(data[:, 0] + data[:, 1]))
n_effs = np.zeros(n_eos_samples)
eos_post = np.zeros(n_eos_samples)
for i in range(n_eos_samples):
    log_weights = data[i * n_m_samples: (i + 1) * n_m_samples, 0]
    n_effs[i] = n_eff_from_log_weights(log_weights)
    eos_post[i] = np.mean(np.exp(log_weights))
eos_post /= np.max(eos_post)
for i in range(n_eos_samples):
    log_weights = data[i * n_m_samples: (i + 1) * n_m_samples, 0]
    #print(i, np.max(log_weights), n_effs[i], eos_post[i])
#mp.hist(n_effs)
#mp.show()
#mp.hist(eos_post / np.max(eos_post))
#mp.show()

# find highest-posterior samples and use their gammas to generate
# a list of mass->lambda mappings
n_map = 10
mkr_map = ['+', '8', 'p', '*', 'h', 'H', '1', '2', '3', '4']
i_map = np.argsort(-data[:, 0])[0: n_map]
fams_map = []
gammas = np.zeros(n_inds)
for i in range(n_map):
    gammas_i = data[i_map[i], 2: 6]
    #print(data[i_map[i], 0])
    if not np.array_equal(gammas_i, gammas):
        gammas = gammas_i
        fam = lal_inf_sd_gammas_fam(gammas)
        m_max_eos = lalsim.SimNeutronStarMaximumMass(fam) / m_sol_kg
    fams_map.append(fam)

# plot mass-lambda relations for all sampled EOSs
n_grid = 100
lambdas = np.zeros(n_grid)
for i in range(n_eos_samples):
#for i in range(100):
    gammas = data[i * n_m_samples, 2: 6]
    fam = lal_inf_sd_gammas_fam(gammas)
    m_max_eos = lalsim.SimNeutronStarMaximumMass(fam) / m_sol_kg
    ms = np.linspace(m_min_ns, m_max_eos * 0.99999999, n_grid)
    for j in range(n_grid):
        lambdas[j] = lal_inf_sd_gammas_mass_to_lambda(fam, ms[j])
    if i == 0:
        mp.plot(ms, lambdas, color='k', alpha=0.05, label='prior draws')
    else:
        mp.plot(ms, lambdas, color='k', alpha=0.05)
ms = np.linspace(m_min_ns, m_max_ns, n_grid)
mp.plot(ms, dd2_lambda_from_mass(ms), 'r--', label='truth')
rect = mpp.Rectangle([m_min_ns, 0.0], m_max_ns - m_min_ns, 4500.0, \
                     facecolor='none', edgecolor='C1', ls=':', \
                     zorder=10, label='per-object prior')
mp.gca().add_patch(rect)
mp.xlabel(r'$m_{\rm ns}\,[m_\odot]$')
mp.ylabel(r'$\Lambda$')
mp.gca().grid(False)
leg = mp.legend(loc='upper right', handlelength=3.0, frameon=False)
for line in leg.get_lines():
    line.set_linewidth(lw)
    line.set_alpha(1.0)
mp.savefig(osp.join(outdir, base_label + '_prior_mass_lambda_relations.pdf'), \
           bbox_inches='tight')
mp.close()

# read injections from file
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

# loop over targets to read in the merger NS mass-lambda posteriors
#n_targets = len(target_ids)
samples = []
truths = []
post_at_true_m_ns = []
m_ns_like_support = []
post_at_true_m_l_ns = []
m_l_ns_posts = []
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
    #print(osp.join(outdir, res_file))
    if not osp.exists(osp.join(outdir, res_file)):
        skip[i] = True
        samples.append(None)
        truths.append(None)
        post_at_true_m_ns.append(None)
        m_ns_like_support.append(None)
        post_at_true_m_l_ns.append(None)
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
                   all_pars['spin_1z'], \
                   all_pars['mass_2'], \
                   all_pars['mass_1_source'], \
                   all_pars['mass_2_source']])

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
        #print(gd_pars)
        if not osp.exists(gd_pars):

            # prevent race conditions: can have all processes trying 
            # to create a symlink simultaneously, with the slow ones
            # finding it's already been created despite not existing 
            # before this if statement
            try:
                os.symlink(template, gd_pars)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise e
            

        # read in samples and fill in derived parameters
        gd_samples = gd.loadMCSamples(gd_root)
        pars = gd_samples.getParams()
        m_1, m_2 = chirp_q_to_comp_masses(pars.chirp_mass, \
                                          pars.mass_ratio)
        gd_samples.addDerived(m_1, name='mass_1', label=r'm_{\rm BH}')
        gd_samples.addDerived(m_2, name='mass_2', label=r'm_{\rm NS}')
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
            post_at_true_m_ns.append(None)
            m_ns_like_support.append(None)
            post_at_true_m_l_ns.append(None)
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
                                   result.posterior.chi_1, \
                                   result.posterior.mass_2, \
                                   result.posterior.mass_1_source, \
                                   result.posterior.mass_2_source]).T
            samples.append(gd.MCSamples(samples=gd_samples, \
                                        names=['a_1', 'mass_1', \
                                               'distance', 'iota', \
                                               'q', 'lambda_2', \
                                               'lambda_tilde', 'chi_1', \
                                               'mass_2', 'mass_1_source', \
                                               'mass_2_source'], \
                                        labels=['a_1', r'm_{\rm BH}', \
                                                distance_label, r'\iota', \
                                                r'm_{\rm NS}/m_{\rm BH}', \
                                                r'\Lambda_{\rm NS}', \
                                                r'\tilde{\Lambda}', \
                                                r'\chi_1', r'm_{\rm NS}', \
                                                r'm_{\rm BH}^{\rm source}', \
                                                r'm_{\rm NS}^{\rm source}'], \
                                        ranges=gd_ranges, weights=weights))
        else:
            gd_samples = np.array([result.posterior.a_1, \
                                   result.posterior.mass_1, \
                                   delta_distance, \
                                   result.posterior.iota, \
                                   result.posterior.mass_ratio, \
                                   result.posterior.lambda_2, \
                                   result.posterior.lambda_tilde, \
                                   result.posterior.spin_1z, \
                                   result.posterior.mass_2, \
                                   result.posterior.mass_1_source, \
                                   result.posterior.mass_2_source]).T
            samples.append(gd.MCSamples(samples=gd_samples, \
                                        names=['a_1', 'mass_1', \
                                               'distance', 'iota', \
                                               'q', 'lambda_2', \
                                               'lambda_tilde', 'chi_1', \
                                               'mass_2', 'mass_1_source', \
                                               'mass_2_source'], \
                                        labels=['a_1', r'm_{\rm BH}', \
                                                distance_label, r'\iota', \
                                                r'm_{\rm NS}/m_{\rm BH}', r'\Lambda_{\rm NS}', \
                                                r'\tilde{\Lambda}', \
                                                r'\chi_1', r'm_{\rm NS}', \
                                                r'm_{\rm BH}^{\rm source}', \
                                                r'm_{\rm NS}^{\rm source}'], \
                                        ranges=gd_ranges, weights=weights))

    # extract m_ns and m_ns-Lambda_ns GetDist posteriors. use the 
    # former to determine the range of m_ns over which the 
    # likelihood has support, which we'll use for sampling EOSs.
    # we'll use the 2D posteriors in the EOS inference itself.
    m_ns_post = samples[-1].get1DDensity('mass_2')
    m_ns_post_grid = m_ns_post.Prob(m_ns_grid)
    i_support = np.where(m_ns_post_grid > support_thresh)
    if i_support[0][0] == 0:
        i_support_min = 0
    else:
        i_support_min = i_support[0][0] - 1
    if i_support[0][-1] == len(m_ns_grid) - 1:
        i_support_max = len(m_ns_grid) - 1
    else:
        i_support_max = i_support[0][-1] + 1
    m_ns_like_support.append(np.array([m_ns_grid[i_support_min], \
                                       m_ns_grid[i_support_max]]))
    post_at_true_m_ns.append(m_ns_post.Prob(truths[-1][9])[0])
    m_l_ns_post = samples[-1].get2DDensity('mass_2', 'lambda_2', \
                                           normalized=False)
    post_at_true_m_l_ns.append(m_l_ns_post(truths[-1][9], \
                                           truths[-1][6])[0, 0])
    m_l_ns_posts.append(m_l_ns_post)


# snr-ordered colouring
n_targets = len(samples)
cm = mpcm.get_cmap('plasma')
cols = [mpc.rgb2hex(cm(x)) for x in np.linspace(0.2, 0.8, n_targets)[::-1]]


# numerical integration tests
n_grid = 1000 # @TODO: test this out. 100, 1000 and 10000 agree pretty well
lambdas = np.zeros(n_grid)
likes = np.zeros(n_grid)
eos_log_post = np.zeros(n_eos_samples)
for i in range(n_eos_samples):
#for i in range(100):
#for i in i_map:

    # extract gammas and use to define mass grid
    gammas = data[i * n_m_samples, 2: 6]
    #gammas = data[i, 2: 6]
    fam = lal_inf_sd_gammas_fam(gammas)
    m_max_eos = lalsim.SimNeutronStarMaximumMass(fam) / m_sol_kg
    ms = np.linspace(m_min_ns, m_max_eos * 0.99999999, n_grid)

    # calculate lambdas on that grid
    for j in range(n_grid):
        lambdas[j] = lal_inf_sd_gammas_mass_to_lambda(fam, ms[j])

    # loop over targets performing integrals
    for k in range(n_targets):

        for j in range(n_grid):

            likes[j] = m_l_ns_posts[k](ms[j], lambdas[j])[0, 0]

        # correct for -ve likelihoods and do trapezoid integration
        neg_like = likes < 0.0
        likes[neg_like] = 0.0
        integral = np.trapz(likes, ms)
        eos_log_post[i] += np.log(integral)

        # @TODO: scipy quad?


n_row, n_col = 2, 2
fig, axes = mp.subplots(n_row, n_col, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        i_ind = j * 2 + i
        axes[j, i].plot(data[0:: n_m_samples, 2 + i_ind], eos_log_post, ',')
        axes[j, i].set_xlabel(r'$\gamma_' + '{:d}'.format(i_ind) + r'$')
        axes[j, i].set_ylabel(r'$\log{\rm P}$')

print(eos_log_post[0: 10])

fig.savefig('test.pdf')

exit()


# tweak output filename
if use_polychord:
    base_label = 'pc_' + base_label
if imp_sample:
    base_label = base_label + '_imp_sample'
if use_polychord and use_weighted_samples:
    base_label = base_label + '_weighted_samples'

# generate figure and plot!
n_col = 8
n_row = int(np.ceil(n_targets / float(n_col)))
#n_row = 2
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

        # plot mass constraint
        g.plot_1d(samples[i], 'mass_2', color=cols[i], \
                  ax=axes[i_y, i_x], normalized=True)
        label = r'$\rho=' + '{:4.1f}'.format(target_snrs[i]) + '$'
        axes[i_y, i_x].text(0.95, 0.95, label, ha='right', va='top', \
                            transform=axes[i_y, i_x].transAxes)
        axes[i_y, i_x].axvline(truths[i][9], color='C1', ls='--')
        axes[i_y, i_x].grid(False)
        axes[i_y, i_x].set_xlim(m_min_ns * 0.99, m_max_ns * 1.01)
        
        # indicate range of significant posterior support
        axes[i_y, i_x].axvspan(m_min_ns * 0.99, m_ns_like_support[i][0], \
                               color='lightgrey')
        axes[i_y, i_x].axvspan(m_ns_like_support[i][1], m_max_ns * 1.01, \
                               color='lightgrey')

        # overlay histogram of mass samples
        axes[i_y, i_x].hist(data[:, n_inds + 2 + i], density=True, histtype='step')

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
fig.savefig(osp.join(outdir, base_label + '_m_ns_constraints_eos_samples.pdf'), \
            bbox_inches='tight')


# ...and a lambda vs m_ns plot...
fig, axes = mp.subplots(n_row, n_col, figsize=(20, height))
g = gdp.get_single_plotter()
i_x = 0
i_y = 0
for i in range(n_row * n_col):
    
    if i < n_targets:

        # watch out for unfinished runs
        if skip[i]:
            fig.delaxes(axes[i_y, i_x])
            i_x += 1
            if i_x == axes.shape[1]:
                i_x = 0
                i_y += 1
            continue

        # plot distance and inclination constraints
        g.plot_2d(samples[i], 'mass_2', 'lambda_2', colors=[cols[i]], \
                  ax=axes[i_y, i_x], filled=True)
        label = r'$\rho=' + '{:4.1f}'.format(target_snrs[i]) + '$'
        axes[i_y, i_x].text(0.95, 0.95, label, ha='right', va='top', \
                            transform=axes[i_y, i_x].transAxes)
        axes[i_y, i_x].plot([truths[i][9]], [truths[i][6]], \
                            marker='+', color='k')
        axes[i_y, i_x].grid(False)
        axes[i_y, i_x].set_xlim(m_min_ns * 0.75, m_max_ns)
        axes[i_y, i_x].set_ylim(0.0, 4500)
        
        # indicate range of significant posterior support
        axes[i_y, i_x].axvspan(m_min_ns * 0.75, m_ns_like_support[i][0], \
                               color='lightgrey')
        axes[i_y, i_x].axvspan(m_ns_like_support[i][1], m_max_ns * 1.01, \
                               color='lightgrey')

        # show highest-posterior samples
        for j in range(n_map):
            m_sample = data[i_map[j], n_inds + 2 + i]
            lambda_sample = lal_inf_sd_gammas_mass_to_lambda(fams_map[j], \
                                                             m_sample)
            if lambda_sample > 4500.0:
                axes[i_y, i_x].plot([m_sample], [4450.0], \
                                    marker='^', color='k', markersize=1.0, \
                                    markeredgecolor='none')
                axes[i_y, i_x].plot([m_sample], [4450.0], \
                                    marker='${:d}$'.format(j), color='darkblue', \
                                    markersize=2.5, markeredgecolor='none')
            elif lambda_sample < 0.0:
                axes[i_y, i_x].plot([m_sample], [50.0], \
                                    marker='v', color='k', markersize=1.0, \
                                    markeredgecolor='none')
                axes[i_y, i_x].plot([m_sample], [50.0], \
                                    marker='${:d}$'.format(j), color='darkblue', \
                                    markersize=2.5, markeredgecolor='none')
            else:
                axes[i_y, i_x].plot([m_sample], [lambda_sample], \
                                    marker='.', color='k', markersize=1.0, \
                                    markeredgecolor='none')
                axes[i_y, i_x].plot([m_sample], [lambda_sample], \
                                    marker='${:d}$'.format(j), color='darkblue', \
                                    markersize=2.5, markeredgecolor='none')
        
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
fig.savefig(osp.join(outdir, base_label + '_lambda_m_ns_constraints_eos_samples.pdf'), \
            bbox_inches='tight')




