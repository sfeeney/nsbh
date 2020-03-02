#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J nsbh
#SBATCH -p CORES24
#SBATCH --array=1-24
#SBATCH --exclusive=user

module purge
source /share/apps/anaconda/3-2019.03/etc/profile.d/conda.sh
conda activate condagw3
module load rocks-openmpi_ib
time python sim_nsbh_analysis.py $SLURM_ARRAY_TASK_COUNT $SLURM_ARRAY_TASK_ID
