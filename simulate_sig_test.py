from getopt import getopt
import pickle
import sys
import multiprocessing
import numpy as np
import pandas as pd
# import seaborn as sns
from scipy.special import logit, expit
from scipy.stats import uniform, norm, bernoulli
from statsmodels.stats.proportion import proportions_ztest
# from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
from modeltools import mcmc_diagnostics, create_summary_stat
from downcast import downcast_df
import jax
from pymc.sampling_jax import sample_numpyro_nuts
from time import time


# python3 simulate_sig_test.py --p_diff 0.055 --n_raters "(20,100)" --scores_per_r 38 --n_sims 1 --out "test"


if __name__ == "__main__":
    
# Calculate missing variable given two of three variables: n_raters, scores_per_r, total_scores, 
    def calculate_raters(n_raters=None, scores_per_r=None, total_scores=None):
    #     Calculate if missing one and only one argument
        count_none = sum([n_raters ==None, scores_per_r ==None, total_scores ==None])
        assert(count_none == 1)

        if n_raters == None:
            n_raters = total_scores//scores_per_r + 1

        if scores_per_r == None:
            scores_per_r = total_scores//n_raters + 1

        if total_scores == None:
            total_scores = n_raters*scores_per_r

        return n_raters, scores_per_r, total_scores
  

    # Create simulation settings template
    def generate_sim_settings(n_sims, p_diff, n_raters=None, scores_per_r=None, total_scores=None):

        # Checking only 2 of 3 score and rater variables is declared
        count_none = sum([n_raters == None, scores_per_r == None, total_scores == None])
        assert count_none == 1, "There should be 2 score/rater variables declared"

        # Sampling uniform random variables
        if type(p_diff) == tuple:
            col_p_diff = uniform.rvs(loc = p_diff[0], scale=p_diff[1]-p_diff[0], size=n_sims)
        else:
            col_p_diff = np.array([p_diff]*n_sims)

        if type(n_raters) == tuple:
            col_n_raters = np.random.randint(n_raters[0], high=n_raters[1], size=n_sims)
        else:
            col_n_raters = np.array([n_raters]*n_sims)

        if type(scores_per_r) == tuple:
            col_scores_per_r = np.random.randint(scores_per_r[0], high=scores_per_r[1], size=n_sims)
        else:
            col_scores_per_r = np.array([scores_per_r]*n_sims)

        if type(total_scores) == tuple:
            col_total_scores = np.random.randint(total_scores[0], high=total_scores[1], size=n_sims)
        else:
            col_total_scores = np.array([total_scores]*n_sims)

        # Calculating remaining column
        if n_raters == None:
            col_n_raters = (col_total_scores-1)//col_scores_per_r + 1
        elif scores_per_r == None:
            col_scores_per_r = (col_total_scores-1)//col_n_raters + 1
        elif total_scores == None:
            col_total_scores = col_scores_per_r * col_n_raters
        else:
            raise Exception("How did you even get this exception? Should've been impossible, but congratulations")

        df = pd.DataFrame(
            np.array([col_p_diff, col_n_raters, col_scores_per_r, col_total_scores]).T,
            columns=["p_diff", "n_raters", "scores_per_r", "total_scores"])

        df = df.astype({
            "p_diff":float,
            "n_raters":np.uint16,
            "scores_per_r":np.uint16,
            "total_scores":np.uint16
        })

        return df
    
    
    # Simulate scores
    def simulate_scores(model, p_diff=0.08, n_raters=40, scores_per_r=40, n_trials=1_000):
    
        def resample(all_ids, param, size, bound=0.1):
            # resampling raters and topics such that effects sum to 0.

            s = model["summary_stat"][model["summary_stat"]["param"]==param].copy(deep=True)

            if param == "za":
                s[["a", "b"]] = s["param_num"].str.split(", ", expand=True)
                s["param_num"] = (s["a"].astype(int)*50 + s["b"].astype(int)).astype(str)

            mean_sum = 9999
            while mean_sum < -bound or mean_sum > bound:
                ids = np.random.choice(all_ids, size=size, replace=True)
                mean_sum = sum([s[(s["param_num"]==str(i))]["mean"].item() for i in ids])

            return ids

        
        ps_data = pd.DataFrame(columns=["trial_id", "sim_cordel_id", "sim_topic_id", "sim_rater_id", 
                                        "cordel_id", "topic_id", "rater_id"], dtype=np.int16)

        for trial_id in range(n_trials):

            # data template
            sim_data = pd.DataFrame(columns=["trial_id", "cordel_id", "topic_id", "rater_id"])  

            # Raters in this simulation
            raters = resample(data["rater_id"].unique(), param="zr", size=n_raters, bound=1)


            # Topics in this simulation (topic_cordel_ids index values)
            sim_topics_0 = resample(range(len(topic_cordel_ids)), param="za", size=50, bound=1)
            sim_topics_1 = resample(range(len(topic_cordel_ids)), param="za", size=50, bound=1)
            sim_topics = np.concatenate((sim_topics_0, sim_topics_1))

            # Count of scores for each topic
            counts = np.zeros(100)

            for sim_rater_id, rater in enumerate(raters):
            #     Set the probability. Topics with fewer samples have higher probability
                counts = counts-counts.min()+1
                p = 1/counts**20
                p = p/p.sum()

            #     Sample according to probability
                rated_topics = np.random.choice(range(100), size=scores_per_r, replace=False, p=p)
                rated_topics_idx = sim_topics[rated_topics]
                counts[rated_topics] += 1

            #     Append topics to simulation
                d=topic_cordel_ids.loc[rated_topics_idx, ["topic_id", "cordel_id"]]
                d["sim_rater_id"]=sim_rater_id
                d["sim_topic_id"]=rated_topics
                d["rater_id"]=rater

                sim_data = pd.concat([sim_data, d], axis="rows", ignore_index=True)

        #     Adding one topic/rater interaction into df
            sim_data["trial_id"] = trial_id
            sim_data.loc[sim_data["sim_topic_id"].isin(range(0,50)),["sim_cordel_id"]] = 0
            sim_data.loc[sim_data["sim_topic_id"].isin(range(50,100)),["sim_cordel_id"]] = 1
        #     sim_data = pd.merge(sim_data, topic_counts[["cordel_id", "topic_id", "sim_cordel_id"]]
        #                         ,on=["cordel_id", "topic_id"], how="left")
            sim_data=sim_data.astype(np.int16)

        #     Appending interaction to ds.
            ps_data = pd.concat([ps_data, sim_data], ignore_index=True)

#         print(f"Completed simulating topic/rater interactions in {time() - startt:.2f}s")

    #     Simulating Scores
        pymc_model = model["model"]
        trace = model["trace"].copy()

        # Calculating proposed logodds means
        # model1 = model0 + p
        # https://www.wolframalpha.com/input?i=solve+for+x+and+y%2C+x%2By%3Dc%2C+1%2F%281%2Be%5E-x%29-1%2F%281%2Be%5E-y%29%3Dp
        mean_model_logodds = model["summary_stat"][model["summary_stat"]["param"]=="mu"]["mean"].mean()
        c = 2*mean_model_logodds
        C = np.exp(-c)
        det = p_diff**2-2*C*(p_diff**2-2)+(C**2)*(p_diff**2)
        quad = (-p_diff*(C+1)+det**0.5)/(2*(p_diff+1))
        proposed_model1_mean = -np.log(quad)
        proposed_model0_mean = c-proposed_model1_mean

        # Setting trace of cordel 0 and cordel 1 to proposed values
        trace.posterior["mu"].loc[dict(mu_dim_0=0)] = proposed_model0_mean
        trace.posterior["mu"].loc[dict(mu_dim_0=1)] = proposed_model1_mean

        sim_scores = pd.DataFrame(columns=["trial_id", "sim_cordel_id", "sim_topic_id", "sim_rater_id", "cordel_id", "topic_id", "rater_id", "intrusion", ]
                           ,dtype=np.int16)

    # TODO: add chain options
        for trial_id in range(n_trials):
            # Setting data containing rater/topic interaction
            sim_data = ps_data[ps_data["trial_id"]==trial_id]
            sim_rater_array = np.array(sim_data["rater_id"], dtype=int)
            topic_array = np.array([sim_data["cordel_id"], sim_data["topic_id"]], dtype=int)
            cordel_array = np.array(sim_data["sim_cordel_id"], dtype=int)

            # Running simulation
            with pymc_model:
                pm.set_data({
                    "raters":sim_rater_array, 
                    "topics":topic_array, 
                    "cordels":cordel_array})
                postrr_sim=pm.sample_posterior_predictive(trace.posterior.sel(
                    {"chain":[0], "draw":[np.random.randint(n_trials) if n_trials==1 else trial_id]})
                    ,predictions=True, progressbar=False, random_seed=np.random.randint(2**20))

            # Adding results to sim_scores
            s = (postrr_sim.predictions.to_dataframe().reset_index()
                  .rename(columns={"s":"intrusion"}))
            this_sim_scores = pd.concat([sim_data.reset_index(drop=True)
                                         ,s["intrusion"]], axis="columns").astype(np.int16)
            sim_scores = pd.concat([sim_scores, this_sim_scores], axis="index", ignore_index=True)
        return sim_scores
    
        
    # Applies proportions z test to find p value that model1 > model0
    def propz_pval(scores):
        # Utests whether the two distributions in the scores are statisticaly significant
        # returns p value of topic model 1>topic model 0

        sum_df = scores.groupby(["sim_cordel_id"]).agg({"intrusion":"sum"}).reset_index()
        cordel0_sums = sum_df[sum_df["sim_cordel_id"]==0]["intrusion"]
        cordel1_sums = sum_df[sum_df["sim_cordel_id"]==1]["intrusion"]

        count_df = scores.groupby(["sim_cordel_id"]).agg({"intrusion":"count"}).reset_index()
        cordel0_counts = count_df[count_df["sim_cordel_id"]==0]["intrusion"]
        cordel1_counts = count_df[count_df["sim_cordel_id"]==1]["intrusion"]

        _, p_val = proportions_ztest([cordel0_sums, cordel1_sums],
                                     [cordel0_counts, cordel1_counts],
                                     alternative="smaller")

        return p_val

    # Applies bayesian hypothesis test to find p value that model1 > model0
    def bht_pval(sample, n_chains):
        # sample = scores[scores["trial_id"]==0]

    # Bayesian hypothesis tests whether the two distributions in the sample are statisticaly significant
    # Setting up numpy arrays for pymc
    # Only 2 models and 1 corpus in simulation
        corpus_array = np.array([0]*len(sample))
        n_corpora = 1

        model_array = np.array(sample["sim_cordel_id"])
        n_models = sample["sim_cordel_id"].nunique()

        cordel_array = np.array(sample["sim_cordel_id"])
        n_cordels = sample["sim_cordel_id"].nunique()

        topic_array = np.array([sample["sim_cordel_id"], sample["sim_topic_id"]])
        n_topics = sample["sim_topic_id"].nunique()

        rater_array = np.array(sample["sim_rater_id"])
        n_raters = sample["sim_rater_id"].nunique()

        score_array = np.array(sample["intrusion"])


        # Model and MCMC specifications
        empirical_mean = logit(0.75)
        r_lambda = 2
        t_lambda = 1
        t_sigma = 1
        # cm_lambda = 2
        # cm_sigma = 1
        mu_sigma = 1

        glm = {"model":pm.Model()}

        # Rater, Topic, Cordel model

        glm["model"] = pm.Model()
        with glm["model"]:
            # Hyperparameter priors
            raters = pm.Data("raters", rater_array, mutable=True, dims="obs_id")
            topics = pm.Data("topics", topic_array, mutable=True, dims=["cordel", "topic"])
            cordels = pm.Data("cordels", cordel_array, mutable=True, dims="obs_id")

            sigma_r = pm.Exponential("sigma_r", lam=r_lambda)
            zr = pm.Normal("zr",mu=0, sigma=1, shape=n_raters)
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
            c_diff = pm.Deterministic("c_diff", c_mean.reshape([n_cordels,1]) - c_mean.reshape([1,n_cordels]), dims="obs_id")

            if SAMPLE_JAX:
                glm["trace"]=sample_numpyro_nuts(chains=n_chains, random_seed=np.random.randint(2**20))#, chain_method="vectorized")
            else:
                glm["trace"]=pm.sample(chains=n_chains, random_seed=np.random.randint(2**20))

        n_negatives = (glm["trace"].posterior["c_diff"].sel({"obs_id":1, "c_diff_dim_1":0}) < 0).sum().item()

        return  n_negatives/len(sample)

    
# Simulates and tests scores for significance
    def simulate_sig_tests(n_sims, p_diff, n_raters=None, scores_per_r=None, 
                       total_scores=None, trials_per_sim=1, np_random_state=None):
    # Bug: Random_state only reproduces scores. Does not reproduce bayesian inference.
    # Reproduce usage: set np_random_state to index of simulation/trial (sim_id*trials_per_sim + trial_id)

        if np_random_state!=None:
            n_sims=1
            trials_per_sim=1

        # Checking only 2 of 3 score and rater variables is declared
        count_none = sum([n_raters == None, scores_per_r == None, total_scores == None])
        assert(count_none == 1)

        # Generate settings for each simulation
        settings_df = generate_sim_settings(n_sims=n_sims, p_diff=p_diff, n_raters=n_raters, 
                                            scores_per_r=scores_per_r, total_scores=total_scores)

        for sim_id in range(n_sims):
            # Reading settings
            p_diff = settings_df.loc[sim_id, "p_diff"].item()
            n_raters = settings_df.loc[sim_id, "n_raters"].item()
            scores_per_r = settings_df.loc[sim_id, "scores_per_r"].item()
            total_scores = settings_df.loc[sim_id, "total_scores"].item()

            # Reading or dumping np.random.RandomState
            if np_random_state==None:
                with open(f"data/{out}/{sim_id*trials_per_sim}.pickle", "wb") as f:
                    pickle.dump(np.random.get_state(), f)
            else:
                with open(f"data/{out}/{np_random_state}.pickle", "rb") as f:
                    np.random.set_state(pickle.load(f))

            scores = simulate_scores(
                glm_rater_topic_cordel,
                p_diff=p_diff,
                n_raters=n_raters,
                scores_per_r=scores_per_r,
                n_trials=trials_per_sim)

            for trial_id in range(trials_per_sim):
                if trial_id != 0:
                    with open(f"data/{out}/{sim_id*trials_per_sim+trial_id}.pickle", "wb") as f:
                            pickle.dump(np.random.get_state(), f)

                sim_results = pd.DataFrame(
                    [[sim_id, p_diff, n_raters, scores_per_r, total_scores, trial_id, 
                     propz_pval(scores[scores["trial_id"]==trial_id]),
                     bht_pval(scores[scores["trial_id"]==trial_id], n_chains=n_chains)]],
                    columns=["sim_id", "p_diff", "n_raters",  "scores_per_r", "total_scores", "trial_id", 
                             "propz_pval", "bht_pval"])

                sim_results = sim_results.astype({
                    "sim_id":int,
                    "trial_id":int,
                    "p_diff":float,
                    "n_raters":np.uint16,
                    "scores_per_r":np.uint16, 
                    "total_scores":np.uint16,
                    "propz_pval":float, 
                    "bht_pval":float,
                })


                if np_random_state==None:
                    with open(f"data/simulations/{out}.csv", mode="a") as f:
                        f.write(sim_results.to_csv(None, index=False, header=False))
                    print(f"Simulation:{sim_id}, Trial:{trial_id}, Cummulative time:{time() - start_time:.2f}")
                else:
                    return scores, sim_results

        return
    
    
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
        if opt == "--p_diff": p_diff = convert_range(value, t=float)
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

    assert(count_none == 1), "Please specify two of: n_raters, scores_per_r, total_scores"

    assert out!=None, "Please specify a valid out file"


    # ====================== Runing Simulation ====================== #
    # Setting numpy seed
    np.random.seed(seed)
    
    # GPU setting
    SAMPLE_JAX = True
    
    # Time
    start_time = time()
    
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
    n_raters = data["rater_id"].nunique()

    score_array = np.array(data["intrusion"])

    # Adding cordel id to topic_ids dataframe
    topic_cordel_ids = pd.merge(topic_ids, cordel_ids, on=["corpus", "model"], how="left")


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
        zr = pm.Normal("zr",mu=0, sigma=1, shape=n_raters)
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
    
    
    simulate_sig_tests(
        p_diff = p_diff,
        n_raters = n_raters,
        scores_per_r = scores_per_r,
        total_scores = total_scores,
        n_sims = n_sims,
        trials_per_sim=trials_per_sim
    )
