import sys
from getopt import getopt
import numpy as np
import pandas as pd
from downcast import downcast_df
from time import time

# python3 create_sim_settings.py --n_raters "(20,100)" --scores_per_r 38 --n_sims 10 --sim_name "test"

if __name__ == "__main__":
    
    # Create simulation settings template
    def generate_sim_settings(n_sims, trials_per_sim, p_diff, n_raters=None, scores_per_r=None, total_scores=None):

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
            np.array([range(n_sims), [trials_per_sim]*n_sims, col_p_diff, 
                      col_n_raters, col_scores_per_r, col_total_scores]).T,
            columns=["sim_id", "trials_per_sim", "p_diff", 
                     "n_raters", "scores_per_r", "total_scores"])
        
        df = df.astype({
            "sim_id":int,
            "trials_per_sim":int,
            "p_diff":float,
            "n_raters":np.uint16,
            "scores_per_r":np.uint16,
            "total_scores":np.uint16
        })
        
        return df
    
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
        "sim_name=",
        "chain_method="])

    # Default values
    p_diff=0.055
    n_raters=None
    scores_per_r=None
    total_scores=None
    n_sims=1000
    trials_per_sim=1
    seed=42
    sim_name=None
    chain_method = "vectorized"

    for opt, value in options:
        if opt == "--p_diff": p_diff = convert_range(value, t=float)
        elif opt == "--n_raters": n_raters = convert_range(value, t=int)
        elif opt == "--scores_per_r": scores_per_r = convert_range(value, t=int)
        elif opt == "--total_scores": total_scores = convert_range(value, t=int)
        elif opt == "--n_sims": n_sims = int(value.strip())
        elif opt == "--trials_per_sim": trials_per_sim = int(value.strip())
        elif opt == "--seed": seed = int(value.strip())
        elif opt == "--sim_name": sim_name = value.strip()
        elif opt == "--chain_method": chain_method = value.strip()

    print(f"""
    p_diff={p_diff}, {type(p_diff)}
    n_raters={n_raters}, {type(n_raters)}
    scores_per_r={scores_per_r}, {type(scores_per_r)}
    total_scores={total_scores}, {type(total_scores)}
    n_sims={n_sims}, {type(n_sims)}
    trials_per_sim={trials_per_sim}
    seed={seed}, {type(seed)}
    sim_name={sim_name}, {type(sim_name)}
    chain_method={chain_method}, {type(chain_method)}
    """)

    # Checking that only of three rater/score is none
    count_none = sum([n_raters==None, scores_per_r==None, total_scores==None])
    assert(count_none == 1), "Please specify two of: n_raters, scores_per_r, total_scores"

    assert sim_name!=None, "Please specify the simulation name"

    # ====================== Simulation ====================== #
    # Number of processes created
    N_PROCESSES = 6
        
    # Generate settings for each simulation
    settings_df = generate_sim_settings(n_sims=n_sims, trials_per_sim=trials_per_sim, p_diff=p_diff, 
                                        n_raters=n_raters, scores_per_r=scores_per_r, total_scores=total_scores)
    
    settings_df.to_csv(
        f"data/{sim_name}/sim_settings.csv",
        index=False)
    
    settings_df = settings_df.rename(columns={"trials_per_sim":"trials"})
    
    total = len(settings_df)
    start = 0
    for i in range(N_PROCESSES):
        if i < total%N_PROCESSES:
            end = start + total//N_PROCESSES + 1
        else:
            end = start + total//N_PROCESSES
            
        settings_df[start:end].to_csv(
            f"data/{sim_name}/sim_settings_{i}.csv",
            index=False)
        
        start = end
        
    # Create Header file
    with open(f"data/simulations/{sim_name}.csv", "w") as f:
        f.write("sim_id,trial_id,p_diff,n_raters,cores_per_r,total_scores,propz_pval,bht_pval\n")