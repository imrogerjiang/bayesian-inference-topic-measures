import pandas as pd
import numpy as np
from downcast import downcast_df
from matplotlib import pyplot as plt
import arviz as az

def plot_prior_postrr(prior_pred, postrr_pred, data, target="s"):
    # Prior
    # Plotting distribution of scores simulated from prior
    prior_samples = prior_pred.prior_predictive.sel(chain=0)
    postrr_samples = postrr_pred.posterior_predictive.sel(chain=0)

    # Manipulating samples to plot
    df_prior, na_s = downcast_df(prior_samples[target].to_dataframe().reset_index())
    df_postrr, na_s = downcast_df(postrr_samples[target].to_dataframe().reset_index())

    # Joining rater/topic/model/corpus information
    data1 = data.copy()
    data1[f"{target}_dim_0"] = data1.index
    df_prior1 = pd.merge(df_prior, data1, on=target+"_dim_0", how="left")
    df_postrr1 = pd.merge(df_postrr, data1, on=target+"_dim_0", how="left")

    # Aggregating "1s" and counts by topic
    prior_agg = df_prior1.groupby(["draw", "topic_id"]).agg({target:["sum", "count"]}).reset_index()
    postrr_agg = df_postrr1.groupby(["draw", "topic_id"]).agg({target:["sum", "count"]}).reset_index()
    obs_agg = df_prior1[df_prior1["chain"]==0].groupby(["draw", "topic_id"]).agg({"intrusion":["sum", "count"]}).reset_index()

    # Calculating topic probabilities from sums and counts
    prior_topic_prob = prior_agg[("s","sum")]/prior_agg[("s","count")]
    postrr_topic_prob = postrr_agg[("s","sum")]/postrr_agg[("s","count")]
    obs_topic_prob = obs_agg[("intrusion","sum")]/obs_agg[("intrusion","count")]

    # Setting up plot
    plot, ax = plt.subplots(1, 1,figsize=(10,6))
    ax = az.plot_kde(np.array(prior_topic_prob), bw=0.05)
    ax.get_lines()[0].set_color("orange")
    ax.get_lines()[0].set_linestyle("--")
    ax = az.plot_kde(np.array(postrr_topic_prob), bw=0.05)
    ax.get_lines()[1].set_color("green")
    ax.get_lines()[1].set_linestyle("--")
    ax = az.plot_kde(np.array(obs_topic_prob), bw=0.05)
    ax.set_title("Prior dist: number of 1's per topic")
    ax.set_ylabel("Density")
    ax.set_xlabel("p")
    ax.legend(ax.get_lines(), ["Prior", "Posterior", "Observed"])

    plt.plot()

def create_summary_stat(trace):
    summary_stat = az.summary(trace, round_to=4).reset_index()

    # Creating parameter and parameter number columns
    summary_stat["param"] = summary_stat["index"].str.split("[").str[0]
    summary_stat["param_num"] = summary_stat["index"].str.split("[").str[1].str[:-1]
    summary_stat["param"] = summary_stat["param"].astype("category")
    summary_stat["param_num"] = summary_stat["param_num"].astype("category")
    return summary_stat[["param", "param_num"]+list(summary_stat.columns[1:-2])]
    

def mcmc_diagnostics(trace, summary_stat):
    # Selecting parameters with highest rhat and lowest ess
    highest_rhat = summary_stat.sort_values(by="r_hat", ascending=False).head(1)
    lowest_ess = summary_stat.sort_values(by="ess_bulk", ascending=True).head(1)
    # If clause for NaN parameter number
    if highest_rhat["param_num"].isna().item():
        highest_rhat = [highest_rhat["param"].item(), None, highest_rhat["r_hat"].item()]
    else:
        highest_rhat = [highest_rhat["param"].item(), int(highest_rhat["param_num"].item()), highest_rhat["r_hat"].item()]

    if lowest_ess["param_num"].isna().item():
        lowest_ess = [lowest_ess["param"].item(), None, lowest_ess["ess_bulk"].item()]
    else: 
        lowest_ess = [lowest_ess["param"].item(), int(lowest_ess["param_num"].item()), lowest_ess["ess_bulk"].item()]

    print("========================== trace diagnostics ==========================")
    print("Divergent transitions")
    print(pd.DataFrame(trace.sample_stats["diverging"]).T.sum(axis="rows"))
    print("\n")
    print("Variable with highest rhat")
    print(highest_rhat)
    print("\n")
    print("Variable with lowest effective sample size")
    print(lowest_ess)

    # Rhat, ESS scatter
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot()
    ax1.scatter(summary_stat["ess_bulk"], summary_stat["r_hat"])
    ax1.set_title("rhat VS ESS")
    ax1.set_xlabel("ESS")
    ax1.set_ylabel("rhat")

    # Trace of worst rhat
    if highest_rhat[1] is None: 
        ax3 = az.plot_trace(trace.posterior[highest_rhat[0]], kind="trace")
    else:
        ax3 = az.plot_trace(trace.posterior[highest_rhat[0]].sel(**{highest_rhat[0]+"_dim_0":highest_rhat[1]}), kind="trace")
    ax3[0][0].set_title(f"Posterior worst rhat: {highest_rhat}")
    ax3[0][1].set_title("trace")

    # Traceof worst ess

    if lowest_ess[1] is None: 
        ax4 = az.plot_trace(trace.posterior[lowest_ess[0]], kind="trace")
    else:
        ax4 = az.plot_trace(trace.posterior[lowest_ess[0]].sel(**{lowest_ess[0]+"_dim_0":lowest_ess[1]}), kind="trace")
    ax4[0][0].set_title(f"Posterior worst ess: {lowest_ess}")
    ax4[0][1].set_title("trace")

    # Trank of worst rhat
    if highest_rhat[1] is None: 
        ax5 = az.plot_rank(trace.posterior[highest_rhat[0]],kind="vlines",
                 vlines_kwargs={'lw':0}, marker_vlines_kwargs={'lw':3})
    else:
        ax5 = az.plot_rank(trace.posterior[highest_rhat[0]].sel(**{highest_rhat[0]+"_dim_0":highest_rhat[1]}),kind="vlines",
                 vlines_kwargs={'lw':0}, marker_vlines_kwargs={'lw':3})
    ax5.set_title(f"Trank worst rhat: {highest_rhat}")

    # Trank of worst ESS
    if lowest_ess[1] is None: 
        ax6 = az.plot_rank(trace.posterior[lowest_ess[0]],kind="vlines",
                 vlines_kwargs={'lw':0}, marker_vlines_kwargs={'lw':3})
    else:
        ax6 = az.plot_rank(trace.posterior[lowest_ess[0]].sel(**{lowest_ess[0]+"_dim_0":lowest_ess[1]}),kind="vlines",
                 vlines_kwargs={'lw':0}, marker_vlines_kwargs={'lw':3})
    ax6.set_title(f"Trank worst ess: {lowest_ess}")

    plt.show()