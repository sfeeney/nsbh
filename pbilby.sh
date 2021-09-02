#!/bin/bash
#SBATCH --job-name=NSBH
#SBATCH --nodes=10  # CHANGE THIS TO CHANGE THE NUMBER OF NODES
#SBATCH --ntasks-per-node=14  # number of CPUs per node
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --output=nsbh.txt
# source /fred/oz006/nsarin/venv_sstar/bin/activate
export MKL_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_NUM_THREADS=1
export MPI_PER_NODE=14

mpirun python sim_nsbh_analysis_4NS.py
