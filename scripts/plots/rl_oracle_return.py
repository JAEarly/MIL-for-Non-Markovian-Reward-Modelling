import sys; sys.path.append("src"); sys.path.append("bonfire")

import argparse
from lib import tufte
from numpy import array, percentile
from matplotlib.pyplot import show

from dataset import TimerTreasureDataset
from dataset import MovingTreasureDataset
from dataset import KeyTreasureDataset
from dataset import ChargerTreasureDataset

parser = argparse.ArgumentParser()
parser.add_argument("task", type=str)
parser.add_argument("--metric", type=str, default="reward_sum_oracle")
args = parser.parse_args()

fnames = [
    (f"reward_source=oracle-augment_state=True---{args.metric}",  "black" ),
    (f"reward_source=oracle-augment_state=False---{args.metric}", "red"   ),
    (f"reward_source=emb-augment_state=True---{args.metric}",     "purple"),
    (f"reward_source=ins-augment_state=True---{args.metric}",     "blue"  ),
    (f"reward_source=csc-augment_state=True---{args.metric}",     "green" ),
]
y_lims = {
    "timer_treasure":   (-3, 50),
    "moving_treasure":  (0, 100),
    "key_treasure":     (-1, 88),
    "charger_treasure": (-1, 43),
    "lunar_lander":     (0, 63) # (0, 50), (0, 331), (-2, 260), (0, 63)
}

tufte.font_size(7)
ax = tufte.ax(
    w_inches=1.2, # Without LunarLander
    # w_inches=0.965, # With LunarLander
    h_inches=0.8, 
    x_label="Episode Number", 
    y_label="Oracle Return",
    y_lims=y_lims[args.task],
    y_ticks=y_lims[args.task]
)

if False: # NOTE: Not using any more
    if task == "timer_treasure":
        dataset_clz = TimerTreasureDataset
        csv_path = "data/oracle/timertreasure/timertreasure.csv"
    elif task == "moving_treasure":
        dataset_clz = MovingTreasureDataset
        csv_path = "data/oracle/movingtreasure/movingtreasure.csv"
    elif task == "charger_treasure":
        dataset_clz = ChargerTreasureDataset
        csv_path = "data/oracle/chargertreasure/chargertreasure.csv"
    else: raise NotImplementedError()

    _, returns, _, _ = dataset_clz.load_data(csv_path=csv_path)
    percentiles = percentile(array([g.item() for g in returns]), q=[25,50,75])
    print(percentiles)
    ax.scatter([0, 0, 0], percentiles, c="k", s=1)

for fname, colour in fnames:
    tufte.smoothed_time_series(ax,
        csv_name=f"results/rl_training/{args.task}/{fname}",
        radius=(25 if args.task == "lunar_lander" else 10), step=1,
        # q=(5,10,15,20,25,30,35,40,45),
        q=(25,),
        colour=colour,
        shade_alpha=0.2,
    )
# show()
ax.save(f"{args.task}_{args.metric}.svg")
