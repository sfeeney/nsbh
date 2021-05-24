import numpy as np
import numpy.random as npr
import bilby
import bilby.gw.conversion as bc
import matplotlib.pyplot as mp
import matplotlib.cm as mpcm
import matplotlib.colors as mpc
import os
import os.path as osp
import errno
import getdist as gd
import getdist.plots as gdp
import copy
import lalsimulation as lalsim
import sys
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

def lalinf_adiabatic_index(gammas, x):

    '''
    Python version of AdiabaticIndex:
    https://lscsoft.docs.ligo.org/lalsuite/lalinference/_l_a_l_inference_8c_source.html#l02608
    '''

    logGamma = 0.0
    for i in range(len(gammas)):
        logGamma += gammas[i] * (x ** i)
    
    return np.exp(logGamma)

def lalinf_sd_gamma_check(gammas):
    
    '''
    Modified from LALInferenceSDGammaCheck, much faster than below:
    https://lscsoft.docs.ligo.org/lalsuite/lalinference/_l_a_l_inference_8c_source.html#l02564
    '''

    p0 = 4.43784199e-13
    xmax = 12.3081
    pmax = p0 * np.exp(xmax)
    ndat = 500

    # Generating higher density portion of EOS with spectral decomposition
    logpmax = np.log(pmax)
    logp0 = np.log(p0)
    dlogp = (logpmax-logp0) / float(ndat)

    # Calculating pressure and adiabatic index table
    for i in reversed(range(ndat)):

        pdat = np.exp(logp0 + dlogp * i)
        xdat = np.log(pdat / p0)
        adat = lalinf_adiabatic_index(gammas, xdat)
        if adat < 0.6 or adat > 4.5:
            return False

    return True

def lalinf_sd_gamma_check_slow(gammas):
    
    '''
    Python version of LALInferenceSDGammaCheck:
    https://lscsoft.docs.ligo.org/lalsuite/lalinference/_l_a_l_inference_8c_source.html#l02564
    '''
    
    p0 = 4.43784199e-13
    xmax = 12.3081
    pmax = p0 * np.exp(xmax)
    ndat = 500

    pdats = np.zeros(ndat)
    adats = np.zeros(ndat)
    xdats = np.zeros(ndat)

    # Generating higher density portion of EOS with spectral decomposition
    logpmax = np.log(pmax)
    logp0 = np.log(p0)
    dlogp = (logpmax-logp0) / float(ndat)

    # Calculating pressure and adiabatic index table
    for i in range(ndat):

        pdat = np.exp(logp0 + dlogp * i)
        xdat = np.log(pdat / p0)
        adat = lalinf_adiabatic_index(gammas, xdat)

        pdats[i] = pdat
        xdats[i] = xdat
        adats[i] = adat

    for i in range(ndat):

        if adats[i] < 0.6 or adats[i] > 4.5:
            return False

    #mp.plot(xdats, adats)
    #mp.show()

    return True

def lal_inf_eos_physical_check(gammas, verbose=False):
    
    '''
    Python version of LALInferenceEOSPhysicalCheck:
    https://lscsoft.docs.ligo.org/lalsuite/lalinference/_l_a_l_inference_8c_source.html#l02404
    '''

    # apply 0.6 < Gamma(p) < 4.5 constraint
    if not lalinf_sd_gamma_check(gammas):
        
        return False
    
    else:

        # create LAL EOS object
        eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(*gammas)
        
        # ensure mass turnover doesn't happen too soon
        mdat_prev = 0.0
        logpmin = 75.5
        logpmax = np.log(lalsim.SimNeutronStarEOSMaxPressure(eos))
        dlogp = (logpmax - logpmin) / 100.0
        for j in range(4):

            # determine if maximum mass has been found
            pdat = np.exp(logpmin + j * dlogp)
            rdat, mdat, kdat = lalsim.SimNeutronStarTOVODEIntegrate(pdat, eos)
            if mdat <= mdat_prev:
                if verbose:
                    print('rejecting: too few EOS points', gammas)
                return False
            mdat_prev = mdat
        
        # make EOS family, and calculate speed of sound and max 
        # and min mass allowed by EOS
        fam = lalsim.CreateSimNeutronStarFamily(eos)
        min_mass_kg = lalsim.SimNeutronStarFamMinimumMass(fam)
        max_mass_kg = lalsim.SimNeutronStarMaximumMass(fam)
        pmax = lalsim.SimNeutronStarCentralPressure(max_mass_kg, fam)
        hmax = lalsim.SimNeutronStarEOSPseudoEnthalpyOfPressure(pmax, eos)
        vsmax = lalsim.SimNeutronStarEOSSpeedOfSoundGeometerized(hmax, eos)

        # apply constraints on speed of sound and maximum mass
        if vsmax > c_s_max:
            if verbose:
                print('rejecting:', \
                      'sound speed {:4.2f} too high'.format(vsmax), \
                      gammas)
            return False
        if max_mass_kg < ns_mass_max_kg:
            if verbose:
                print('rejecting:', \
                      'max NS mass {:4.2f} too low'.format(max_mass_kg / m_sol_kg), \
                      gammas)
            return False

        return True

def lal_inf_sd_gammas_mass_to_lambda(gammas, mass_m_sol):

    '''
    Modified from LALInferenceSDGammasMasses2Lambdas:
    https://lscsoft.docs.ligo.org/lalsuite/lalinference/_l_a_l_inference_8c_source.html#l02364
    '''

    # create EOS & family
    eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(*gammas)
    fam = lalsim.CreateSimNeutronStarFamily(eos)

    # calculate lambda(m|eos)
    mass_kg = mass_m_sol * m_sol_kg
    rad = lalsim.SimNeutronStarRadius(mass_kg, fam)
    love = lalsim.SimNeutronStarLoveNumberK2(mass_kg, fam)
    comp = big_g * mass_kg / (c ** 2) / rad
    
    return 2.0 / 3.0 * love / comp ** 5


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



# @TODO: incorporate maximum mass per EOS?
#        needs some thinking. there's a maximum mass used in the 
#        initial sampling (which, in turn, is complicated by the 
#        fact we had to use M_c and q): this is the maximum mass
#        specified in the prior. 1) if the EOS permits larger
#        masses than this, should we alter the prior volume and 
#        normalization? 2) if the EOS's maximum mass is lower 
#        than the prior, we need to reject prior draws higher 
#        than the EOS's m_max. should we instead draw from an
#        EOS-specific range each time? 3) do we need to adjust 
#        the prior volume each time?
# @TODO: KDE version
# @TODO: store Lambdas too?
# @TODO: skip duff IMRPhenom run
# @TODO: store log prior separately
# @TODO: parallelize w/ MPI
# @TODO: plot highest-weighted EOS samples on m vs Lambda posterior


# EOS inference settings
n_samples = 5 # 160 # @TODO: update w/ mass samples too
n_m_samples = 10 # 100
m_r_alpha = 0.5 # @TODO: check if needed
n_inds = 4
prior_ranges = [[0.2, 2.0], [-1.6, 1.7], [-0.6, 0.6], [-0.02, 0.02]]
c_s_max = 1.1
ns_mass_max_kg = 1.97 * m_sol_kg

# data sample settings
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
lam_det_test = False
old = False
outdir = 'outdir'
support_thresh = 1.0e-3

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

# set up identical within-chain MPI processes
if use_mpi:
    import mpi4py.MPI as mpi
    n_procs = mpi.COMM_WORLD.Get_size()
    rank = mpi.COMM_WORLD.Get_rank()
elif len(sys.argv) > 1:
    if len(sys.argv) == 3:
        n_procs = int(sys.argv[1])
        rank = int(sys.argv[2])
        if rank > n_procs or rank < 1:
            exit('ERROR: 1 <= rank <= number of processes.')
        rank -= 1    
    else:
        exit('ERROR: please call using ' + \
             '"python sim_nsbh_analysis.py <n_procs> <rank>" format ' + \
             'to specify number of processes and rank without MPI. ' + \
             'NB: rank should be one-indexed.')
else:
    n_procs = 1
    rank = 0

# set rank-specific random seed
npr.seed(221216 + rank * 10)

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

# useful grids in neutron star mass and lambda
m_ns_grid = np.linspace(m_min_ns, m_max_ns, 1000)

# loop over targets to read in the merger NS mass-lambda posteriors
n_targets = len(target_ids)
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


# draw potential SD EOSs
all_gammas = []
gammas = np.zeros(n_inds)
masses = np.zeros(n_targets)
n_samples_tot = n_samples * n_m_samples
pop_samples = np.zeros((n_inds + n_targets, n_samples_tot))
log_weights = np.zeros(n_samples_tot)
log_priors = np.zeros(n_samples_tot)
i = 0
n_rej = 0
while i < n_samples:

    # draw random samples from SD priors
    for j in range(n_inds):

        gammas[j] = npr.uniform(prior_ranges[j][0], \
                                prior_ranges[j][1])

    # check if physical using LAL code, which is much quicker
    if not lal_inf_eos_physical_check(gammas):
        continue

    # calculate maximum mass allowed by EOS: don't allow prior draws 
    # beyond this limit
    eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(*gammas)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    m_max_eos = lalsim.SimNeutronStarMaximumMass(fam) / m_sol_kg

    # loop over mass draws per EOS
    for k in range(n_m_samples):

        # sample neutron star masses from most efficient range. this 
        # is from the greater of m_min_ns and m_ns_like_support[j][0] 
        # (which is automatically satisfied in the calculation of 
        # m_ns_like_support) to the lesser of m_max_eos and 
        # m_ns_like_support[j][1]. the weights must then be adjusted 
        # to account for the restricted prior range used, which 
        # should be m_min_ns -> m_max_eos
        i_tot = i * n_m_samples + k
        for j in range(n_targets):
            m_max_sample = min(m_ns_like_support[j][1], m_max_eos)
            masses[j] = npr.uniform(m_ns_like_support[j][0], m_max_sample)
            #prior_norm_used = 1.0 / (m_max_sample - m_ns_like_support[j][0])
            prior_norm_true = 1.0 / (m_max_eos - m_min_ns)
            #log_weights[i_tot] += np.log(prior_norm_true / prior_norm_used)
            log_priors[i_tot] += np.log(prior_norm_true)

        # store sampled parameters
        pop_samples[0: n_inds, i_tot] = gammas[:]
        pop_samples[n_inds:, i_tot] = masses[:]

        # calculate the tidal deformability for each 
        # sampled mass and add the log-likelihood to the log-weights
        for j in range(n_targets):

            lambda_lal = lal_inf_sd_gammas_mass_to_lambda(gammas, masses[j])
            like = m_l_ns_posts[j](masses[j], lambda_lal)[0, 0]
            if like < 0.0:
                log_weights[i_tot] = log_zero
                continue
            else:
                log_weights[i_tot] += np.log(like)

            # @TODO: make sure -ve weights happen where expected?

    # increment
    i += 1
        

    '''
    # maximum mass
    eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(*gammas)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    m_max = lalsim.SimNeutronStarMaximumMass(fam) / m_sol_kg
    n_grid = 100
    m_grid = np.linspace(1.0, 0.999 * m_max, n_grid)
    r_grid = np.zeros(n_grid)
    for j in range(n_grid):
        r_grid[j] = lalsim.SimNeutronStarRadius(m_grid[j] * m_sol_kg, fam)
        '''

# save to file
fname = 'data/' + label + \
        '_n_det_proc_{:d}_of_{:d}.txt'.format(rank, n_procs)
filename = osp.join(outdir, \
                    base_label + \
                    '_eos_samples_{:d}_of_{:d}.txt'.format(rank, n_procs))
np.savetxt(filename, np.vstack((log_weights, log_priors, pop_samples)).T)


# normalize weights
log_weights = log_weights - np.max(log_weights)
log_weights_norm = np.log(np.sum(np.exp(log_weights)))
log_weights = log_weights - log_weights_norm
weights = np.exp(log_weights)
n_eff = np.exp(-np.sum(weights * log_weights))
print(rank, n_eff)


exit()


# @TODO: might need more compression in data storage
# e.g. only store summed over masses
# but also store lambdas?

