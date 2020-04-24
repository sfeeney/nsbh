import numpy as np
import matplotlib.pyplot as mp

# setup
datadir = 'data/noise_curves/'
files = []

# LVC noise curves from https://dcc.ligo.org/LIGO-T2000012/public
files.append('LVC_AplusDesign.txt')
files.append('LVC_avirgo_O5low_NEW.txt')
files.append('LVC_avirgo_O5high_NEW.txt')
files.append('LVC_kagra_128Mpc.txt')
files.append('LVC_kagra_80Mpc.txt')

# LVC noise curves from bilby's LVC source: 
# https://dcc.ligo.org/LIGO-P1200087-v42/public
files.append('LVC_fig1_aligo_sensitivity.txt')
files.append('LVC_fig1_adv_sensitivity.txt')
files.append('LVC_fig1_kagra_sensitivity.txt')

# bilby noise curves: NB bilby_aplus.txt = bilby_Aplus_asd.txt
files.append('bilby_Aplus_asd.txt')
files.append('bilby_KAGRA_design_psd.txt')
files.append('bilby_aLIGO_ZERO_DET_high_P_psd.txt')
files.append('bilby_AdV_psd.txt')

# read in noise curves
noise_curves = []
for file in files:
	noise_curves.append(np.genfromtxt(datadir + file))

# ALIGO+ plot
fig, axes = mp.subplots(2, 2, figsize=(10, 10))
axes[0, 0].loglog(noise_curves[0][:, 0], noise_curves[0][:, 1], \
				  label='LVC aLIGO+ (AW)', color='C0')
axes[0, 0].loglog(noise_curves[8][:, 0], noise_curves[8][:, 1], \
				  ls='--', label='bilby aLIGO+', color='C3')
axes[0, 0].loglog(noise_curves[5][:, 0], noise_curves[5][:, 5], \
				  label='LVC aLIGO (bilby source)', color='C1')
axes[0, 0].loglog(noise_curves[10][:, 0], np.sqrt(noise_curves[10][:, 1]), \
				  ls='--', label='bilby aLIGO', color='C2')
axes[0, 0].legend(loc='upper right', fontsize=8)

# AdVirgo+ plot
#axes[0, 1].loglog(noise_curves[1][:, 0], noise_curves[1][:, 1], \
#				  label='LVC AdVirgo+ low')
axes[0, 1].loglog(noise_curves[2][:, 0], noise_curves[2][:, 1], \
				  label='LVC AdVirgo+ high (AW)', color='C0')
axes[0, 1].loglog(noise_curves[6][:, 0], noise_curves[6][:, 6], \
				  label='LVC AdVirgo (bilby source)', color='C1')
axes[0, 1].loglog(noise_curves[11][:, 0], np.sqrt(noise_curves[11][:, 1]), \
				  ls='--', label='bilby AdVirgo', color='C2')
axes[0, 1].legend(loc='upper right', fontsize=8)

# KAGRA plot
axes[1, 0].loglog(noise_curves[3][:, 0], noise_curves[3][:, 1], \
				  label='LVC KAGRA (design AW)', color='C0')
axes[1, 0].loglog(noise_curves[4][:, 0], noise_curves[4][:, 1], \
				  label='LVC KAGRA (O4 AW)', color='C3')
axes[1, 0].loglog(noise_curves[7][:, 0], noise_curves[7][:, 5], \
				  label='LVC KAGRA (bilby source)', color='C1')
axes[1, 0].loglog(noise_curves[9][:, 0], np.sqrt(noise_curves[9][:, 1]), \
				  ls='--', label='bilby KAGRA', color='C2')
axes[1, 0].legend(loc='upper right', fontsize=8)

# LIGO India plot
axes[1, 1].loglog(noise_curves[0][:, 0], noise_curves[0][:, 1], \
				  label='LVC IndIGO (AW aLIGO+)', color='C0')
axes[1, 1].loglog(noise_curves[10][:, 0], np.sqrt(noise_curves[10][:, 1]), \
				  ls='--', label='previous assumption', color='C1')
axes[1, 1].legend(loc='upper right', fontsize=8)

# finish off plot
for i in range(2):
	for j in range(2):
		axes[i, j].set_xlim(1e0, 2e4)
		axes[i, j].set_ylim(1e-24, 5e-20)
		if i == 1:
			axes[i, j].set_xlabel('frequency [Hz]')
		else:
			axes[i, j].get_xaxis().set_visible(False)
		if j == 0:
			axes[i, j].set_ylabel('strain noise [1/Hz$^2$]')
		else:
			axes[i, j].get_yaxis().set_visible(False)
fig.subplots_adjust(wspace=0.0, hspace=0.0)
fig.savefig(datadir + 'noise_comparisons.pdf', bbox_inches='tight')

# convert downloaded ASD files to PSD files for bilby to use
for i in range(5):
	noise_curves[i][:, 1] = noise_curves[i][:, 1] ** 2
	filename = files[i].replace('.txt', '_PSD.txt')
	np.savetxt(datadir + filename, noise_curves[i], delimiter=' ', \
			   fmt='%.9e')
