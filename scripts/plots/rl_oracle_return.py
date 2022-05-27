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
args = parser.parse_args()

fnames = [
    ("reward_source=oracle-augment_state=False---reward_sum_oracle", "red"   ),
    ("reward_source=oracle-augment_state=True---reward_sum_oracle",  "black" ),
    # ("reward_source=csc-augment_state=False---reward_sum_oracle",  "orange"),
    ("reward_source=emb-augment_state=True---reward_sum_oracle",   "purple"    ),
    ("reward_source=ins-augment_state=True---reward_sum_oracle",   "blue"   ),
    ("reward_source=csc-augment_state=True---reward_sum_oracle",   "green"  ),
]
y_lims = {
    "timer_treasure":   (-3, 50),
    "moving_treasure":  (0, 100),
    "key_treasure":     (-1, 88),
    "charger_treasure": (-1, 43)
}

tufte.font_size(7)
ax = tufte.ax(
    w_inches=1.2, 
    h_inches=0.8, 
    x_label="Episode Number", 
    y_label="Oracle Return",
    y_lims=y_lims[args.task],
    y_ticks=(0, 50)
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
        radius=10, step=1,
        # q=(5,10,15,20,25,30,35,40,45),
        q=(25,),
        colour=colour,
        shade_alpha=0.2,
    )
# show()
ax.save(f"{args.task}_oracle_return.svg")
