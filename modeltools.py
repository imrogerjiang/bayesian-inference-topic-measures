import pandas as pd
from matplotlib import pyplot as plt
import arviz as az

def mcmc_diagnostics(trace, param):
    rhat = pd.DataFrame(az.rhat(trace)[param], columns=[param]).sort_values(by=param, ascending=False)[:5]
    worst_rhat = rhat.index[0]
    ess = pd.DataFrame(az.ess(trace)[param], columns=[param]).sort_values(by=param)[:5]
    worst_ess = ess.index[0]


    print("========================== Trace diagnostics ==========================")
    print("Divergent transitions")
    print(pd.DataFrame(trace.sample_stats["diverging"]).T.sum(axis="rows"))
    print("\n")
    print("Vars with highest rhat")
    print(rhat)
    print("\n")
    print("5 variables with lowest effective sample size")
    print(ess)

    # Rhat, ESS scatter
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot()
    ax1.scatter(az.ess(trace)[param], az.rhat(trace)[param])
    ax1.set_title("rhat VS ESS")
    ax1.set_xlabel("ESS")
    ax1.set_ylabel("rhat")

    # Trace of worst rhat
    ax3 = az.plot_trace(trace.posterior[param].sel(**{param+"_dim_0":worst_rhat}), kind="trace")
    ax3[0][0].set_title(f"Posterior worst rhat: {param}[{worst_rhat}]")
    ax3[0][1].set_title("Trace")

    # Trace of worst ess
    if worst_rhat != worst_ess:
        ax4 = az.plot_trace(trace.posterior[param].sel(**{param+"_dim_0":worst_ess}), kind="trace")
        ax4[0][0].set_title(f"Posterior worst ess: {param}[{worst_ess}]")
        ax4[0][1].set_title("Trace")

    # Trank of worst rhat
    ax5 = az.plot_rank(trace.posterior[param].sel(**{param+"_dim_0":worst_rhat}),kind="vlines",
                 vlines_kwargs={'lw':0}, marker_vlines_kwargs={'lw':3})
    ax5.set_title(f"Trank worst rhat: {param}[{worst_rhat}]")

    # Trank of worst ESS
    if worst_rhat != worst_ess:
        ax6 = az.plot_rank(trace.posterior[param].sel(**{param+"_dim_0":worst_ess}),kind="vlines",
                     vlines_kwargs={'lw':0}, marker_vlines_kwargs={'lw':3})
        ax6.set_title(f"Trank worst ess: {param}[{worst_ess}]")

    plt.plot()