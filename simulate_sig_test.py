from getopt import getopt
import sys
import multiprocessing
import numpy as np
import pandas as pd
# import seaborn as sns
from scipy.special import logit, expit
from scipy.stats import uniform, norm, bernoulli
# from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
from modeltools import mcmc_diagnostics, create_summary_stat
from downcast import downcast_df
import jax
from pymc.sampling_jax import sample_numpyro_nuts
from time import time

# ====================== Checking Input  ====================== #
def convert_range(string, t=float):
    try:
        return t(string)
    except ValueError:
        nums = string[1:-1].strip().split(",")
        return (t(nums[0].strip()), t(nums[1].strip()))
    
argv = sys.argv[1:]
options, args = getopt(argv, "",[
    "p_diff=",
    "n_raters=",
    "scores_per_r=",
    "total_scores=",
    "n_sims=",
    "trials_per_sim=",
    "seed=",
    "out=",])

# Default values
p_diff=0.055
n_raters=None
scores_per_r=None
total_scores=None
n_sims=1000
trials_per_sim=1
seed=42
out=None

for opt, value in options:
    if opt == "--p_diff": 
        p_diff = convert_range(value, t=float)
        print(p_diff)
    elif opt == "--n_raters": n_raters = convert_range(value, t=int)
    elif opt == "--scores_per_r": scores_per_r = convert_range(value, t=int)
    elif opt == "--n_sims": n_sims = int(value.strip())
    elif opt == "--trials_per_sim": trials_per_sim = int(value.strip())
    elif opt == "--seed": seed = int(value.strip())
    elif opt == "--out": out = value.strip()


        
print(f"""
p_diff={p_diff}, {type(p_diff)}
n_raters={n_raters}, {type(n_raters)}
scores_per_r={scores_per_r}, {type(scores_per_r)}
total_scores={total_scores}, {type(total_scores)}
n_sims={n_sims}, {type(n_sims)}
trials_per_sim={trials_per_sim}, {type(trials_per_sim)}
seed={seed}, {type(seed)}
out={out}, {type(out)}
""")

# Checking that only of three rater/score is none
count_none = sum([n_raters==None, scores_per_r==None, total_scores==None])
assert(count_none == 1)

assert out!=None, "Please specify a valid out file"


print("Number of cpu : ", multiprocessing.cpu_count())

