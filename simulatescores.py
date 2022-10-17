import numpy as np
import pandas as pd
# import xarray as xr
import pymc as pm
# import arviz as az
# import seaborn as sns
# import itertools

from scipy.special import logit, expit
# from scipy.stats import bernoulli, norm, t, skewnorm
# from matplotlib import pyplot as plt
from time import time

# from modeltools import plot_prior_postrr, create_summary_stat, mcmc_diagnostics 
from downcast import downcast_df

def simulate_scores(model, p_diff=0.08, n_raters=40, scores_per_r=40, n_sims=1_000):
    
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

    # Adding cordel id to topic_ids dataframe
    topic_cordel_ids = pd.merge(topic_ids, cordel_ids, on=["corpus", "model"], how="left")
    
    ps_data = pd.DataFrame(columns=["sim_id", "sim_cordel_id", "sim_topic_id", "sim_rater_id", 
                                    "cordel_id", "topic_id", "rater_id"], dtype=np.int16)

    for sim_id in range(n_sims):

        # data template
        sim_data = pd.DataFrame(columns=["sim_id", "cordel_id", "topic_id", "rater_id"])  

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
        sim_data["sim_id"] = sim_id
        sim_data.loc[sim_data["sim_topic_id"].isin(range(0,50)),["sim_cordel_id"]] = 0
        sim_data.loc[sim_data["sim_topic_id"].isin(range(50,100)),["sim_cordel_id"]] = 1
    #     sim_data = pd.merge(sim_data, topic_counts[["cordel_id", "topic_id", "sim_cordel_id"]]
    #                         ,on=["cordel_id", "topic_id"], how="left")
        sim_data=sim_data.astype(np.int16)

    #     Appending interaction to ds.
        ps_data = pd.concat([ps_data, sim_data], ignore_index=True)
    
    print(f"Completed simulating topic/rater interactions in {time() - start_time:.2f}s")
    
#     Simulating Scores
    pymc_model = model["model"]
    trace = model["trace"]

    # Calculating proposed logodds means
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
    
    sim_scores = pd.DataFrame(columns=["sim_id", "sim_cordel_id", "sim_topic_id", "sim_rater_id", "cordel_id", "topic_id", "rater_id", "intrusion", ]
                       ,dtype=np.int16)
    
# TODO: add chain options
    for sim_id in range(n_sims):
        # Setting data containing rater/topic interaction
        sim_data = ps_data[ps_data["sim_id"]==sim_id]
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
                {"chain":[0], "draw":[np.random.randint(n_sims) if n_sims==1 else sim_id]})
                ,predictions=True, progressbar=False, random_seed=np.random.randint(2**20))

        # Adding results to sim_scores
        s = (postrr_sim.predictions.to_dataframe().reset_index()
              .rename(columns={"s":"intrusion"}))
        this_sim_scores = pd.concat([sim_data.reset_index(drop=True)
                                     ,s["intrusion"]], axis="columns").astype(np.int16)
        sim_scores = pd.concat([sim_scores, this_sim_scores], axis="index", ignore_index=True)
    
    print(f"Completed simulating scores in {time() - start_time:.2f}s")
    return sim_scores