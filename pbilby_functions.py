import inspect
import sys

import numpy as np
import afterglowpy as grb
import bilby
from astropy.io import fits
from astropy.table import Table
from scipy.special import gammaln
from scipy.integrate import simps
from schwimmbad import MPIPool
from numpy import linalg
from bilby.core.utils import reflect
from dynesty.utils import unitcheck
from bilby.core.prior import Constraint
import mpi4py
import dynesty
import datetime
import os
import pickle

import time
import dill
from dynesty import NestedSampler
import dynesty.plotting as dyplot
logger = bilby.core.utils.logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sample_rwalk_parallel_with_act(args):
    """ A dynesty sampling method optimised for parallel_bilby

    """

    # Unzipping.
    (u, loglstar, axes, scale, prior_transform, loglikelihood, kwargs) = args
    rstate = np.random
    # Bounds
    nonbounded = kwargs.get("nonbounded", None)
    periodic = kwargs.get("periodic", None)
    reflective = kwargs.get("reflective", None)

    # Setup.
    n = len(u)
    walks = kwargs.get("walks", 50)  # minimum number of steps
    maxmcmc = kwargs.get("maxmcmc", 10000)  # maximum number of steps
    nact = kwargs.get("nact", 10)  # number of act

    accept = 0
    reject = 0
    nfail = 0
    act = np.inf
    u_list = []
    v_list = []
    logl_list = []

    drhat, dr, du, u_prop, logl_prop = np.nan, np.nan, np.nan, np.nan, np.nan
    while len(u_list) < nact * act:
        # Propose a direction on the unit n-sphere.
        drhat = rstate.randn(n)
        drhat /= linalg.norm(drhat)

        # Scale based on dimensionality.
        dr = drhat * rstate.rand() ** (1.0 / n)

        # Transform to proposal distribution.
        du = np.dot(axes, dr)
        u_prop = u + scale * du

        # Wrap periodic parameters
        if periodic is not None:
            u_prop[periodic] = np.mod(u_prop[periodic], 1)
        # Reflect
        if reflective is not None:
            u_prop[reflective] = reflect(u_prop[reflective])

        # Check unit cube constraints.
        if unitcheck(u_prop, nonbounded):
            pass
        else:
            nfail += 1
            if accept > 0:
                u_list.append(u_list[-1])
                v_list.append(v_list[-1])
                logl_list.append(logl_list[-1])
            continue

        # Check proposed point.
        v_prop = prior_transform(u_prop)
        logl_prop = loglikelihood(v_prop)
        if logl_prop >= loglstar:
            u = u_prop
            v = v_prop
            logl = logl_prop
            accept += 1
            u_list.append(u)
            v_list.append(v)
            logl_list.append(logl)
        else:
            reject += 1
            if accept > 0:
                u_list.append(u_list[-1])
                v_list.append(v_list[-1])
                logl_list.append(logl_list[-1])

        # If we've taken the minimum number of steps, calculate the ACT
        if accept + reject > walks:
            act = bilby.core.sampler.dynesty.estimate_nmcmc(
                accept_ratio=accept / (accept + reject + nfail),
                old_act=walks,
                maxmcmc=maxmcmc,
                safety=5,
            )

        # If we've taken too many likelihood evaluations then break
        if accept + reject > maxmcmc:
            logger.warning(
                "Hit maximum number of walks {} with accept={}, reject={}, "
                "nfail={}, and act={}. Try increasing maxmcmc".format(
                    maxmcmc, accept, reject, nfail, act
                )
            )
            break

    # If the act is finite, pick randomly from within the chain
    factor = 0.1
    if len(u_list) == 0:
        logger.warning("No accepted points: returning -inf")
        u = u
        v = prior_transform(u)
        logl = -np.inf
    elif np.isfinite(act) and int(factor * nact * act) < len(u_list):
        idx = np.random.randint(int(factor * nact * act), len(u_list))
        u = u_list[idx]
        v = v_list[idx]
        logl = logl_list[idx]
    else:
        logger.warning(
            "len(u_list)={}<{}: returning the last point in the chain".format(
                len(u_list), int(factor * nact * act)
            )
        )
        u = u_list[-1]
        v = v_list[-1]
        logl = logl_list[-1]

    blob = {"accept": accept, "reject": reject, "fail": nfail, "scale": scale}

    ncall = accept + reject
    return u, v, logl, ncall, blob

def read_saved_state(resume_file, continuing=True):
    """
    Read a saved state of the sampler to disk.

    The required information to reconstruct the state of the run is read from a
    pickle file.

    Parameters
    ----------
    resume_file: str
        The path to the resume file to read

    Returns
    -------
    sampler: dynesty.NestedSampler
        If a resume file exists and was successfully read, the nested sampler
        instance updated with the values stored to disk. If unavailable,
        returns False
    sampling_time: int
        The sampling time from previous runs
    """

    if os.path.isfile(resume_file):
        logger.info("Reading resume file {}".format(resume_file))
        with open(resume_file, "rb") as file:
            sampler = dill.load(file)
            if sampler.added_live and continuing:
                sampler._remove_live_points()
            sampler.nqueue = -1
            sampler.rstate = np.random
            sampling_time = sampler.kwargs.pop("sampling_time")
        return sampler, sampling_time
    else:
        logger.info("Resume file {} does not exist.".format(resume_file))
        return False, 0



def safe_file_dump(data, filename, module):
    """ Safely dump data to a .pickle file

    Parameters
    ----------
    data:
        data to dump
    filename: str
        The file to dump to
    module: pickle, dill
        The python module to use
    """

    temp_filename = filename + ".temp"
    with open(temp_filename, "wb") as file:
        module.dump(data, file)
        print(module)
    os.rename(temp_filename, filename)


def write_current_state(sampler, resume_file, sampling_time):
    """ Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    resume_file: str
        The name of the resume/checkpoint file to use
    sampling_time: float
        The total sampling time in seconds
    """
    print("")
    #print('prior transform',sampler.prior_transform)
    #delete the sample_rwalk_parallel_with_act function before pickle
    print(sampler.evolve_point)
    del sampler.evolve_point
    del sampler.prior_transform
    del sampler.loglikelihood
    sampler.kwargs["sampling_time"] = sampling_time
    # safe_file_dump(sampler, resume_file, dill)
    if dill.pickles(sampler):
        safe_file_dump(sampler, resume_file, dill)
        logger.info("Written checkpoint file {}".format(resume_file))
    else:
        logger.warning(
            "Cannot write pickle resume file! " "Job will not resume if interrupted."
        )


def plot_current_state(sampler, search_parameter_keys, outdir, label):
    labels = [label.replace("_", " ") for label in search_parameter_keys]
    try:
        filename = "{}/{}_checkpoint_trace.png".format(outdir, label)
        fig = dyplot.traceplot(sampler.results, labels=labels)[0]
        fig.tight_layout()
        fig.savefig(filename)
    except (
        AssertionError,
        RuntimeError,
        np.linalg.linalg.LinAlgError,
        ValueError,
    ) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty state plot at checkpoint")
    finally:
        plt.close("all")
    try:
        filename = "{}/{}_checkpoint_run.png".format(outdir, label)
        fig, axs = dyplot.runplot(sampler.results)
        fig.tight_layout()
        plt.savefig(filename)
    except (RuntimeError, np.linalg.linalg.LinAlgError, ValueError) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty run plot at checkpoint")
    finally:
        plt.close("all")
    try:
        filename = "{}/{}_checkpoint_stats.png".format(outdir, label)
        fig, axs = plt.subplots(nrows=3, sharex=True)
        for ax, name in zip(axs, ["boundidx", "nc", "scale"]):
            ax.plot(getattr(sampler, f"saved_{name}"), color="C0")
            ax.set_ylabel(name)
        axs[-1].set_xlabel("iteration")
        fig.tight_layout()
        plt.savefig(filename)
    except (RuntimeError, ValueError) as e:
        logger.warning(e)
        logger.warning("Failed to create dynesty stats plot at checkpoint")
    finally:
        plt.close("all")
