from getopt import getopt
import cloudpickle
import pickle
import sys
import os
import numpy as np
import pandas as pd
# import seaborn as sns
from scipy.special import logit, expit
from scipy.stats import uniform, norm, bernoulli, betabinom
from statsmodels.stats.proportion import proportions_ztest
# from matplotlib import pyplot as plt
import pymc as pm
import arviz as az
from modeltools import mcmc_diagnostics, create_summary_stat
from downcast import downcast_df
import jax
from pymc.sampling_jax import sample_numpyro_nuts
from time import time, sleep
from datetime import timedelta

# Usage python3 perform_sig_test.py --trials_per_sim 1 --optimal_alloc "True" --n_runs 30 --process 0 --sim_name "test"

if __name__ == "__main__":
    
        # Simulate scores
    def simulate_scores(model, p_diff=0.08, n_raters=40, scores_per_r=40, total_scores=None, 
                        trials_per_sim=1_000, seed=42, optimal_allocation=False):

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

        def postrr_var(n_success, total):
            a = n_success+1
            b = total-n_success+1
            return a*b/((a+b+1)*(a+b)**2)
        
        def postrr_p(n_success, total):
            a = n_success+1
            b = total-n_success+1
            return betabinom.pmf(n=1,k=1,a=a,b=b)

        # Setting numpy seed
        np.random.seed(seed)

        # Creating df schema
        ps_data = pd.DataFrame(columns=["trial_id", "sim_cordel_id", "sim_topic_id", "sim_rater_id", 
                                        "cordel_id", "topic_id", "rater_id"], dtype=np.int16)

        for trial_id in range(trials_per_sim):

            # data template
            sim_data = pd.DataFrame(columns=["trial_id", "cordel_id", "topic_id", "rater_id"])  

            # Raters in this simulation
            raters = resample(data["rater_id"].unique(), param="zr", size=n_raters, bound=1)

            # Topics in this simulation (topic_cordel_ids index values)
            sim_topics_0 = resample(range(len(topic_cordel_ids)), param="za", size=50, bound=1)
            sim_topics_1 = resample(range(len(topic_cordel_ids)), param="za", size=50, bound=1)
            sim_topics = np.concatenate((sim_topics_0, sim_topics_1))
            
            # Loop - used to contain uniform sampling algorithm could use cleanup
            # Produces df containing cross product between raters and topics
            for sim_rater_id, rater in enumerate(raters):
                rated_topics = np.array(range(100))

                rated_topics_idx = sim_topics[rated_topics]

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
        for trial_id in range(trials_per_sim):
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
                    {"chain":[0], "draw":[np.random.randint(trials_per_sim) if trials_per_sim==1 else trial_id]})
                    ,predictions=True, progressbar=False, random_seed=np.random.randint(2**20))

            # Adding results to sim_scores
            s = (postrr_sim.predictions.to_dataframe().reset_index()
                  .rename(columns={"s":"intrusion"}))
            trial_sim_scores = pd.concat([sim_data.reset_index(drop=True)
                                         ,s["intrusion"]], axis="columns").astype(np.int16)

            if optimal_allocation:
                # Optimal topic allocation scores
                scores = trial_sim_scores[:0]

                if total_scores == None:
                    total_scores = n_raters*scores_per_r

                # Allocate topics for each rater
                for sim_rater_id in range(n_raters):
                    # Checking if all scores have been allocated
                    if total_scores <= 0:
                        break

                    # Calculating variance for each topic's posterior distribution
                    s = (scores.groupby("sim_topic_id").agg({"intrusion":"sum"})
                         .rename(columns={"intrusion":"sum"}).reset_index())
                    c = (scores.groupby("sim_topic_id").agg({"intrusion":"count"})
                         .rename(columns={"intrusion":"count"}).reset_index())
                    topic_var = pd.merge(s, c, on="sim_topic_id")

                    # Create df with zeros if no data exists
                    if len(topic_var) < 100:
                        missing_topic_ids = [i for i in range(100) if i not in np.array(topic_var["sim_topic_id"])]
                        missings = pd.DataFrame({"sim_topic_id":missing_topic_ids
                                                  ,"sum":[0]*len(missing_topic_ids)
                                                  ,"count":[0]*len(missing_topic_ids)})
                        topic_var = pd.concat([topic_var, missings])

                    # Calculating reduction in variance
                    topic_var["variance"] = postrr_var(topic_var["sum"], topic_var["count"])
                    topic_var["p"] = postrr_p(topic_var["sum"], topic_var["count"])
                    topic_var["var0"] = postrr_var(topic_var["sum"], topic_var["count"]+1)
                    topic_var["var1"] = postrr_var(topic_var["sum"]+1, topic_var["count"]+1)
                    topic_var["expected_var"] = topic_var["p"]*topic_var["var1"]+(1-topic_var["p"])*topic_var["var0"]
                    topic_var["var_diff"] = topic_var["expected_var"]-topic_var["variance"]

                    # Allocating topics
                    allocated_topics = topic_var.sort_values("var_diff", ascending=True)[:scores_per_r]["sim_topic_id"]
                    total_scores -= scores_per_r

                    selected_scores = trial_sim_scores[(trial_sim_scores["sim_rater_id"]==sim_rater_id)&
                                        (trial_sim_scores["sim_topic_id"].isin(allocated_topics))]
                    scores = pd.concat([scores, selected_scores])
            else:
                # Running total of scores
                counts = np.zeros(100)
                scores = trial_sim_scores[:0]
                for sim_rater_id, rater in enumerate(raters):
                    # Set the probability. Topics with fewer samples have higher probability
                    counts = counts-counts.min()+1
                    p = 1/counts**20
                    p = p/p.sum()

                    # Sample according to probability
                    allocated_topics = np.random.choice(range(100), size=scores_per_r, replace=False, p=p)
                    counts[allocated_topics] += 1

                    selected_scores = trial_sim_scores[(trial_sim_scores["sim_rater_id"]==sim_rater_id)&
                                    (trial_sim_scores["sim_topic_id"].isin(allocated_topics))]
                    scores = pd.concat([scores, selected_scores])
                    
            sim_scores = pd.concat([sim_scores, scores], axis="index", ignore_index=True)
            sim_scores.to_csv(f"data/{sim_name}/score_{sim_id}.csv", index=False)
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
    def bht_pval(sample, n_chains, seed=None):
        # sample = scores[scores["trial_id"]==0]
        if seed != None:
            np.random.seed(seed)
            
        # pval incorrect when sim_topic_id  is between (0, 100).
        # Error is due to c_mean adding all 100 "topics" for each cordel even 50 invalid ones
        # Corrects sim_topics of cordel 1 to 0-50
        sample.loc[sample["sim_cordel_id"]==1, "sim_topic_id"]-=50
    
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
            c_diff = pm.Deterministic("c_diff", c_mean[1]- c_mean[0])


            if SAMPLE_JAX:
                glm["trace"]=sample_numpyro_nuts(chains=1, random_seed=np.random.randint(2**20), chain_method=chain_method)
            else:
                glm["trace"]=pm.sample(chains=n_chains, random_seed=np.random.randint(2**20))

        
        n_negatives = (glm["trace"].posterior["c_diff"]<0).sum().item()
        total = glm["trace"].posterior["c_diff"].count().item()

        return  n_negatives/total



    # ====================== Checking Input  ====================== #
    argv = sys.argv[1:]
    options, args = getopt(argv, "",[
        "process=",
        "n_runs=",
        "trials_per_sim=",
        "optimal_alloc=",
        "seed=",
        "sim_name=",
        "chain_method="])

    # Default values
    process_n=None
    n_runs=30
    trials_per_sim=1
    optimal_allocation=False
    seed=42
    sim_name=None
    chain_method = "vectorized"

    for opt, value in options:
        if opt == "--process": process_n = int(value.strip())
        elif opt == "--n_runs": n_runs = int(value.strip())
        elif opt == "--trials_per_sim": trials_per_sim = int(value.strip())
        elif opt == "--optimal_alloc": optimal_allocation = value.strip().lower() == "true"
        elif opt == "--seed": seed = int(value.strip())
        elif opt == "--sim_name": sim_name = value.strip()
        elif opt == "--chain_method": chain_method = value.strip()
            
    print(f"""
    process={process_n}
    n_runs={n_runs}
    trials_per_sim={trials_per_sim}
    optimal_alloc={optimal_allocation}
    seed={seed}
    sim_name={sim_name}
    chain_method={chain_method}
    """)
    
    assert sim_name!=None, "Please specify the simulation name"
    assert process_n!=None, "Please specify the process number"

    
    # File indicates process is running
    with open(f"data/{sim_name}/process_{process_n}_running", "w") as f:
        pass
    
    # ====================== Read Model and setup ====================== #
    # GPU setting
    SAMPLE_JAX = True
    N_PROCESSES = 6

    # Time
    start_time = time()

    # Reading in data
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

    # Reading model
    with open("bayesian_model/glmm.pickle", "rb") as f:
        inferred_glmm = cloudpickle.load(f)

    # ====================== Simulation ====================== #
    
    # Read simulation settings
    sim_settings_file = f"data/{sim_name}/sim_settings_{process_n}.csv"
    sim_settings = pd.read_csv(sim_settings_file)

    for run in range(n_runs):
        
        sim_id = int(sim_settings.iloc[0]["sim_id"])
        p_diff = sim_settings.iloc[0]["p_diff"]
        n_raters = int(sim_settings.iloc[0]["n_raters"])
        scores_per_r = int(sim_settings.iloc[0]["scores_per_r"])
        total_scores = int(sim_settings.iloc[0]["total_scores"])
        trial_id = int(sim_settings.iloc[0]["trial_id"])
        
        # Dumping np.random.RandomState
        with open(f"data/{sim_name}/{sim_id*trials_per_sim + trial_id}.pickle", "wb") as f:
                pickle.dump(np.random.get_state(), f)
        
        run_seed = seed + sim_id*trials_per_sim + trial_id
        scores = simulate_scores(
            inferred_glmm,
            p_diff=p_diff,
            n_raters=n_raters,
            optimal_allocation=optimal_allocation,
            scores_per_r=scores_per_r,
            trials_per_sim=1,
            seed=run_seed
        )

        sim_results = pd.DataFrame(
            [[sim_id, trial_id, p_diff, n_raters, scores_per_r, total_scores, 
             propz_pval(scores), bht_pval(scores, n_chains=1, seed=run_seed), run_seed]],
            columns=["sim_id", "trial_id", "p_diff", "n_raters",  "scores_per_r", "total_scores", 
                     "propz_pval", "bht_pval", "seed"])

        sim_results = sim_results.astype({
            "sim_id":int,
            "trial_id":int,
            "p_diff":float,
            "n_raters":np.uint16,
            "scores_per_r":np.uint16, 
            "total_scores":np.uint16,
            "propz_pval":float, 
            "bht_pval":float,
            "seed":np.uint16,
        })

        # Write results to file
        with open(f"data/simulations/{sim_name}.csv", mode="a") as f:
            f.write(sim_results.to_csv(None, index=False, header=False))

        # Decrementing trials var on file
        # If only 1 row remaining on file, delete file
        if len(sim_settings) == 1:
            os.remove(sim_settings_file)
            break
        # else delete row
        else:
            sim_settings=sim_settings[1:]
            sim_settings.to_csv(sim_settings_file, index=False)


        print(f"===== Simulation-{sim_id}, Trial-{trial_id}, Cumulative time-{timedelta(seconds=time() - start_time)} =====")
    
#     Deleting running file to start new process.
    os.remove(f"data/{sim_name}/process_{process_n}_running")