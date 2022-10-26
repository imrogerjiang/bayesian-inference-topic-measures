from getopt import getopt
import cloudpickle
import sys
import numpy as np
import pandas as pd
# import seaborn as sns
from scipy.special import logit#, expit
# from scipy.stats import uniform, norm, bernoulli
# from statsmodels.stats.proportion import proportions_ztest
# from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
from modeltools import mcmc_diagnostics, create_summary_stat
from downcast import downcast_df
import jax
from pymc.sampling_jax import sample_numpyro_nuts
# from time import time, sleep, timedelta

if __name__=="__main__":
    
    # ====================== Checking Input  ====================== #
    argv = sys.argv[1:]
    options, args = getopt(argv, "",[
        "seed=",
        "chain_method="])

    # Default values
    seed=42
    chain_method = "vectorized"

    for opt, value in options:
        if opt == "--seed": seed = int(value.strip())
        elif opt == "--chain_method": chain_method = value.strip()

    print(f"""
    seed={seed}
    chain_method={chain_method}
    """)

    # ====================== Read file and setup ====================== #
    # Setting numpy seed
    np.random.seed(seed)
    
    # GPU setting
    SAMPLE_JAX = True
    N_PROCESSES = 6
    
    # Time
    raw_data = pd.read_csv("data/unit_level_ratings.csv",index_col = 0)
    raw_data = raw_data.sort_values(by=["corpus", "model", "topic"])
    
    # Creating identifier for each corpus, model, and topic
    # Identifier is unique for topic 
    corpus_ids = (raw_data.groupby(["corpus"], as_index=False)
        .agg({"intrusion":"count"})
        .drop(columns="intrusion"))
    corpus_ids["corpus_id"] = corpus_ids.index

    model_ids = (raw_data.groupby(["model"], as_index=False)
        .agg({"intrusion":"count"})
        .drop(columns="intrusion"))
    model_ids["model_id"] = model_ids.index

    cordel_ids = (raw_data.groupby(["corpus", "model"], as_index=False)
        .agg({"intrusion":"count"})
        .drop(columns="intrusion"))
    cordel_ids["cordel_id"] = cordel_ids.index 

    topic_ids = (raw_data.groupby(["corpus", "model", "topic"], as_index=False)
        .agg({"intrusion":"count"})
        .drop(columns="intrusion"))
    topic_ids["topic_id"] = topic_ids["topic"].astype(np.int16)

    rater_ids = (raw_data.groupby(["corpus", "rater"], as_index=False)
        .agg({"intrusion":"count"})
        .drop(columns="intrusion"))
    rater_ids["rater_id"] = rater_ids.index 

    d1 = pd.merge(raw_data, corpus_ids, on=["corpus"], how="left")
    d2 = pd.merge(d1, model_ids, on=["model"], how="left")
    d3 = pd.merge(d2, cordel_ids, on=["corpus","model"], how="left")
    d4 = pd.merge(d3, rater_ids, on=["corpus", "rater"], how="left")
    data = pd.merge(d4, topic_ids, on=["corpus", "model", "topic"], how="left")
    data = data[["corpus_id", "model_id", "cordel_id", "topic_id", "rater_id", "intrusion", "confidence"]]
    data, na_s = downcast_df(data)
    
    # Setting up numpy arrays for pymc
    corpus_array = np.array(data["corpus_id"])
    n_corpora = data["corpus_id"].nunique()

    model_array = np.array(data["model_id"])
    n_models = data["model_id"].nunique()

    cordel_array = np.array(data["cordel_id"])
    n_cordels = data["cordel_id"].nunique()

    topic_array = np.array([data["cordel_id"], data["topic_id"]])
    n_topics = data["topic_id"].nunique()

    rater_array = np.array(data["rater_id"])
    obs_n_raters = data["rater_id"].nunique()

    score_array = np.array(data["intrusion"])

    # Adding cordel id to topic_ids dataframe
    topic_cordel_ids = pd.merge(topic_ids, cordel_ids, on=["corpus", "model"], how="left")


    # ====================== Infer generative model params ====================== #
    # Model and MCMC specifications
    n_chains = 1
    empirical_mean = logit(0.75)
    r_lambda = 2
    t_lambda = 1
    t_sigma = 1
    # cm_lambda = 2
    # cm_sigma = 1
    mu_sigma = 1

    glm_rater_topic_cordel = {"model":pm.Model()}

    # Rater, Topic, Cordel model
    glm_rater_topic_cordel["model"] = pm.Model()
    with glm_rater_topic_cordel["model"]:
        # Hyperparameter priors
        raters = pm.Data("raters", rater_array, mutable=True, dims="obs_id")
        topics = pm.Data("topics", topic_array, mutable=True, dims=["cordel", "topic"])
        cordels = pm.Data("cordels", cordel_array, mutable=True, dims="obs_id")

        sigma_r = pm.Exponential("sigma_r", lam=r_lambda)
        zr = pm.Normal("zr",mu=0, sigma=1, shape=obs_n_raters)
        sigma_a = pm.Exponential("sigma_a", lam=t_lambda)
        za = pm.Normal("za",mu=0, sigma=t_sigma, shape=(n_cordels, n_topics)) 
        mu = pm.Normal("mu",mu=empirical_mean, sigma=mu_sigma, shape=n_cordels)

        s = pm.Bernoulli(
                "s", 
                p=pm.math.invlogit(
                    mu[cordels]+
                    za[topics[0],topics[1]]*sigma_a+
                    zr[raters]*sigma_r),
                observed=score_array, 
                dims="obs_id")

        c_mean = pm.Deterministic("c_mean", 
                                  pm.math.invlogit(mu + (za.T*sigma_a)).mean(axis=0), 
                                  dims="obs_id")

        if SAMPLE_JAX:
            glm_rater_topic_cordel["trace"]=sample_numpyro_nuts(chains=n_chains, random_seed=seed)#, chain_method="vectorized")
        else:
            glm_rater_topic_cordel["trace"]=pm.sample(chains=n_chains, random_seed=seed)
    
    glm_rater_topic_cordel["summary_stat"] = create_summary_stat(glm_rater_topic_cordel["trace"])
    
    with open("bayesian_model/glmm.pickle", "wb") as f:
        cloudpickle.dump(glm_rater_topic_cordel, f)
        