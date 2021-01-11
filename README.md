# Adventures in inferring neutron-star-black-hole binary parameters

[![arXiv](https://img.shields.io/badge/arXiv-2012.06593-green.svg)](https://arxiv.org/abs/2012.06593)

Code to simulate and analyse populations of neutron-star-black-hole binary mergers. There are a number of Python files included in this repository, but the basic workflow is

 - `generate_nsbh_sample.py`: generate sample of simulated mergers (saves parameters and RNG states required to recreate exact noise realisation)
 - `sim_nsbh_analysis.py`: per-merger bilby analysis (use script following `hypatia_no_mpi_slurm.sh` to accelerate using HPC) 
 - `n_bar_det_farming.py` / `n_bar_det_processing.py`: calculate expected number of detections as function of cosmology (SLURM-able as above)
 - `population_posteriors.py`: optionally fit GW distance likelihoods with Gaussian mixtures, and sample from population and cosmology posteriors

Note the following dependencies:

 - [AstroPy](https://www.astropy.org/)
 - [Bilby](https://lscsoft.docs.ligo.org/bilby/index.html)
 - [GetDist](http://getdist.readthedocs.io/en/latest/intro.html)
 - [pomegranate](https://pomegranate.readthedocs.io/en/latest/)
 - [PyStan](https://pystan.readthedocs.io/en/latest/)
 - [pypolychord](https://github.com/PolyChord/PolyChordLite)
 - [scikit-learn](http://scikit-learn.org/stable/install.html)

Authors: Stephen Feeney, Hiranya Peiris, Samaya Nissanke and Daniel Mortlock (and Andrew Williamson for ns_eos_aw.py!).
