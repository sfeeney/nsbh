#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -J nsbh
#SBATCH -p CORES24

module purge
source /share/apps/anaconda/3-2019.03/etc/profile.d/conda.sh
conda activate condagw2
module load rocks-openmpi_ib
time python bilby_nsbh_injection.py
