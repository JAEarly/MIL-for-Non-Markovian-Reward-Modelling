import sys; sys.path.append("src")

from lib import tufte
from lib.env import HoloNav
import pandas as pd
from numpy import array, expand_dims, append
from matplotlib.pyplot import ioff
from matplotlib.patches import Rectangle

from rl_training.maps import maps
from oracles.rl.timertreasure import TimerTreasureOracle


TASK = "timer_treasure"
RUN_NAME = "kind-bush-55"
EP_NUM = 120


if TASK == "timer_treasure":
    oracle = TimerTreasureOracle()
    map_ = maps["timertreasure"]

tufte.font_size(7)
tufte.line_width(0.369)

# Render map
env = HoloNav(map_)
env.ax = tufte.ax(
    w_inches=0.782, 
    h_inches=0.782,
    x_lims=(0, 1), 
    y_lims=(0, 1)
)
env.ax.axis("off")
env.render_map(); ioff()

df = pd.read_csv(f"observations/{RUN_NAME}.csv")
ep = df[df["ep"] == EP_NUM]
bag = ep[["x", "y"]].values
rewards = ep["reward"].values

# Compute reward and return error
oracle.reset()
oracle_rewards = array([oracle(instance) for instance in bag])
reward_error = rewards - oracle_rewards
return_error = reward_error.sum()
print(rewards.sum(), oracle_rewards.sum(), return_error)

high = max(abs(reward_error))
low = -high

# Plot trajectory
cbar_ax = tufte.coloured_2d_plot(env.ax,
    data=expand_dims(bag, 0),
    # colour_by="time"
    colour_by=expand_dims(reward_error, 0),
    cmap="coolwarm_r",
    cmap_lims=(low, high)
)

# Plot plot reward error time series
ax = tufte.ax(
    w_inches=1.2, 
    h_inches=0.8,
    x_lims=(0,100),
    x_ticks=(0, 50, 100)
)
ax.step(range(len(rewards)+1), append(rewards, rewards[-1]), where="post", c="k")
ax.step(range(len(rewards)+1 ), append(oracle_rewards, oracle_rewards[-1]), where="post", c="k", ls=":")
cmap = tufte.sns_cmap("coolwarm_r")
c = [cmap((e - low) / (high - low)) for e in reward_error]
for i in range(len(reward_error)):
    ax.add_artist(Rectangle((i, oracle_rewards[i]), width=1, height=reward_error[i], color=c[i], linewidth=0))

tufte.show()
env.ax.save("env.svg")
ax.save("error.svg")
cbar_ax.save("cbar.svg")
cbar_ax.axis("off")
cbar_ax.save("cbar.png")
