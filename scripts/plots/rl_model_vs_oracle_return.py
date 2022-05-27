import sys; sys.path.append("src"); sys.path.append("pytorch_mil")

from lib import tufte
import pandas as pd
from matplotlib.pyplot import show

from dataset import KeyTreasureDataset


task = "timer_treasure"

fnames = [
    "reward_source=model-augment_state=True---reward_sum",
    "reward_source=model-augment_state=True---reward_sum_oracle",
]

data = \
    pd.read_csv(f"results/rl_training/{task}/{fnames[0]}.csv").values.T \
    # - \
    # pd.read_csv(f"results/rl_training/{task}/{fnames[1]}.csv").values.T

tufte.font_size(7)
ax = tufte.ax(
    w_inches=1.2, 
    h_inches=0.8, 
    x_label="Episode Number", 
    y_label="Oracle Return",
    y_lims=(-20, 51),
    y_ticks=(0, 100),
)

tufte.coloured_2d_plot(ax, data=data, plot_type="scatter", cmap="tab10")

# For identifying outliers
# repeat = 9
# range_ = range(120, 121)
# ax.scatter(range_, data[repeat,range_], s=30)
# ax.plot([0, 400], [-7.888529099524021, -7.888529099524021])

print(ax.get_ylim())

# show()

# ax.save("fig.svg")

ax.axis("off")
ax.save("fig.png", dpi=300)
