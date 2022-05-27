import argparse
from lib.rlutils.experiments.wandb import get


parser = argparse.ArgumentParser()
parser.add_argument("username", type=str)
parser.add_argument("task", type=str)
parser.add_argument("reward_source", type=str)
parser.add_argument("--augment_state", type=int, default=1)
parser.add_argument("--path", type=str, default="results/rl_training")
args = parser.parse_args()

for data in get(
    project_name=f"{args.username}/mil-rm_{args.task}",
    metrics=[
        "reward_sum_oracle",
        # "reward_sum", # NOTE: For CSC only, to go in appendix
    ],
    filters=[{"reward_source": args.reward_source,  "augment_state": bool(args.augment_state)}]
    ):
    for m in data:
        assert data[m]["df"].shape == (400, 10)
        assert not data[m]["df"].isnull().values.any()
        data[m]["df"].to_csv(f"{args.path}/{args.task}/{data[m]['fname']}", index=False)
