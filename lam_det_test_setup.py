import numpy as np
import pickle

def chirp_mass(m_1, m_2):

    return (m_1 * m_2) ** 0.6 / (m_1 + m_2) ** 0.2

def dd2_lambda_from_mass(m):
    return 1.60491e6 - 23020.6 * m**-5 + 194720. * m**-4 - 658596. * m**-3 \
        + 1.33938e6 * m**-2 - 1.78004e6 * m**-1 - 992989. * m + 416080. * m**2 \
        - 112946. * m**3 + 17928.5 * m**4 - 1263.34 * m**5

# settings
to_store = ['simulation_id', 'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', \
            'spin2x' , 'spin2y', 'spin2z', 'distance', 'inclination', \
            'coa_phase', 'polarization', 'longitude', 'latitude', \
            'geocent_end_time', 'geocent_end_time_ns']
simulation_id = 3991
mass1 = 6.09096
mass2 = 1.439077
spin1x = -0.07754489
spin1y = 0.07397693
spin1z = -0.1961679
spin2x = -0.0101551
spin2y = 0.01806639
spin2z = -0.01132604
distance = 89.83986
inclination = 0.584028
coa_phase = 0.5873603
polarization = 2.522222
longitude = 2.297854
latitude = -0.4085953
geocent_end_time = 1382266734
geocent_end_time_ns = 27086306
lambda2 = dd2_lambda_from_mass(mass2)

# vary spin magnitude at fixed orientation
spin1_mag = np.sqrt(spin1x ** 2 + spin1y ** 2 + spin1z ** 2)
spin1_mags = np.array([0.0, 0.45, 0.675, 0.9])
spin1xs = spin1x * spin1_mags / spin1_mag
spin1ys = spin1y * spin1_mags / spin1_mag
spin1zs = spin1z * spin1_mags / spin1_mag

# vary spin orientation at fixed magnitude: permute components
# so the spin is strongest towards the x and y directions
spin1xs_or = np.array([spin1z, spin1y])
spin1ys_or = np.array([spin1x, spin1z])
spin1zs_or = np.array([spin1y, spin1x])

# vary inclination
iotas = np.linspace(0.0, 1.0, 5) * np.pi

# vary mass ratio at fixed distance
m_1s = np.array([2.2, 2.7, 3.2, 3.7, 4.7, 5.2]) * mass2

# vary mass ratio at fixed SNR by changing distance. this is 
# approximate, as I'm using  a linear form for the distance-
# redshift relation
m_c = chirp_mass(mass1, mass2)
m_cs = chirp_mass(m_1s, mass2)
h_0 = 67.36
c = 2.998e5
dists = 1.0 / (m_c / m_cs * (1.0 / distance + h_0 / c) - h_0 / c)

# fudge for the last thing that results in better SNRs
dists = distance / 72.75508970795582 * \
		np.array([59.93680323121011, 63.10721655668162, \
				  65.23454677489202, 68.97930910995535, \
				  74.92976154599194, 76.66923271035213])

# save parameters to file
label = 'lam_det_test'
fmt = '{:s},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},' + \
      '{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:d},' + \
      '{:d},{:.9e},{:.9e},{:.9e}'
with open('data/' + label + '.txt', 'w') as f:
    j = 0
    s = 'sim_inspiral:simulation_id:{:d}'.format(j)
    j += 1
    f.write('#' + ','.join(to_store) + ',lambda_2,remnant_mass,snr')
    f.write('\n' + \
            fmt.format(s, mass1, mass2, spin1x, spin1y, spin1z, \
                       spin2x, spin2y, spin2z, distance, inclination, \
                       coa_phase, polarization, longitude, latitude, \
                       geocent_end_time, geocent_end_time_ns, lambda2, \
                       0.000013, 43.784000))
    for i in range(len(spin1_mags)):
        s = 'sim_inspiral:simulation_id:{:d}'.format(j)
        j += 1
        f.write('\n' + \
                fmt.format(s, mass1, mass2, spin1xs[i], spin1ys[i], \
                           spin1zs[i], spin2x, spin2y, spin2z, distance, \
                           inclination, coa_phase, polarization, longitude, \
                           latitude, geocent_end_time, geocent_end_time_ns, \
                           lambda2, 0.000013, 43.784000))
    for i in range(len(spin1xs_or)):
        s = 'sim_inspiral:simulation_id:{:d}'.format(j)
        j += 1
        f.write('\n' + \
                fmt.format(s, mass1, mass2, spin1xs_or[i], spin1ys_or[i], \
                           spin1zs_or[i], spin2x, spin2y, spin2z, distance, \
                           inclination, coa_phase, polarization, longitude, \
                           latitude, geocent_end_time, geocent_end_time_ns, \
                           lambda2, 0.000013, 43.784000))
    for i in range(len(iotas)):
        s = 'sim_inspiral:simulation_id:{:d}'.format(j)
        j += 1
        f.write('\n' + \
                fmt.format(s, mass1, mass2, spin1x, spin1y, \
                           spin1z, spin2x, spin2y, spin2z, distance, \
                           iotas[i], coa_phase, polarization, longitude, \
                           latitude, geocent_end_time, geocent_end_time_ns, \
                           lambda2, 0.000013, 43.784000))
    for i in range(len(m_1s)):
        s = 'sim_inspiral:simulation_id:{:d}'.format(j)
        j += 1
        f.write('\n' + \
                fmt.format(s, m_1s[i], mass2, spin1x, spin1y, \
                           spin1z, spin2x, spin2y, spin2z, distance, \
                           inclination, coa_phase, polarization, longitude, \
                           latitude, geocent_end_time, geocent_end_time_ns, \
                           lambda2, 0.000013, 43.784000))
    for i in range(len(m_1s)):
        s = 'sim_inspiral:simulation_id:{:d}'.format(j)
        j += 1
        f.write('\n' + \
                fmt.format(s, m_1s[i], mass2, spin1x, spin1y, \
                           spin1z, spin2x, spin2y, spin2z, dists[i], \
                           inclination, coa_phase, polarization, longitude, \
                           latitude, geocent_end_time, geocent_end_time_ns, \
                           lambda2, 0.000013, 43.784000))

# save random states to file
np.random.seed(221216)
rng_states = [np.random.get_state()] * j
with open('data/' + label + '_rng_states.bin', 'wb') as f:
    pickle.dump(rng_states, f)
