import sys; sys.path.append("src"); sys.path.append("bonfire")

import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from oracles.rl.lunar_lander import LunarLanderOracle


parser = argparse.ArgumentParser()
parser.add_argument("reward_source", type=str)
args = parser.parse_args()

oracle = LunarLanderOracle()

dir_name = f"results/rl_training/lunar_lander/trajectories/{args.reward_source}"
run_names = sorted([x[:-4] for x in os.listdir(dir_name) if x[-4:] == ".csv"], key=lambda x: int(x.split(".")[0].split("-")[-1]))
decomposed_return = np.zeros((len(run_names), 800, 4))

for r, run_name in enumerate(run_names):
    df = pd.read_csv(f"{dir_name}/{run_name}.csv")
    ep_starts = np.argwhere(df["time"].values == 0)[1:].flatten()
    bags = np.split(df[LunarLanderOracle.input_names].values, ep_starts)
    for i, bag in tqdm(enumerate(bags[:800]), desc=run_name, total=800):
        oracle.reset()
        for instance in bag:
            reward = oracle(instance)
            decomposed_return[r,i,3] += reward % 1 # R_shaping
            if reward >= 1.:
                if oracle.internal_state[0] < oracle.land_duration: decomposed_return[r,i,0] += 1. # R_pad
                else:
                    decomposed_return[r,i,1] += 1. # R_no_contact
                    if reward >= 2.: decomposed_return[r,i,2] += 1.

for c, component_name in enumerate(["R_pad", "R_no_contact", "R_hover", "R_shaping"]):
    df = pd.DataFrame(decomposed_return[:,:,c].T, columns=run_names)
    df.to_csv(f"results/rl_training/lunar_lander/reward_source={args.reward_source}-augment_state=True---{component_name}.csv", index=False)
