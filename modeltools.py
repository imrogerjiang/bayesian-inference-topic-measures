import pandas as pd
from matplotlib import pyplot as plt
import arviz as az

def mcmc_diagnostics2(trace, summary_stat):
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