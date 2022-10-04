"""
"Generate toy datasets for evaluating MIL models with temporal dependencies."

Example args:
    switch --num_bags=100 --min_bag_size=10 --max_bag_size=20 --seed=0
"""
import sys; sys.path.append("src"); sys.path.append("bonfire")

import argparse

from oracles.synthetic.dial import DialOracle
from oracles.synthetic.push_switch import PushSwitchOracle
from oracles.synthetic.toggle_switch import ToggleSwitchOracle
from oracles.rl.keytreasure import KeyTreasureOracle
from oracles.rl.timertreasure import TimerTreasureOracle
from oracles.rl.movingtreasure import MovingTreasureOracle
from oracles.rl.chargertreasure import ChargerTreasureOracle
from oracles.rl.lunar_lander import LunarLanderOracle


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=str)
    parser.add_argument("--num_bags", type=int, default=5000)
    parser.add_argument("--min_bag_size", type=int, default=10)
    parser.add_argument("--max_bag_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str)
    args, _ = parser.parse_known_args()
    oracle_name = args.oracle

    if oracle_name == "push_switch":
        parser.add_argument("--push_proba", type=float, default=0.2)
        args = parser.parse_args()
        PushSwitchOracle.generate_dataset(args.num_bags, args.min_bag_size, args.max_bag_size, args.seed,
                                          args.save_path, push_proba=args.push_proba)
    elif oracle_name == "toggle_switch":
        parser.add_argument("--toggle_proba", type=float, default=0.2)
        args = parser.parse_args()
        ToggleSwitchOracle.generate_dataset(args.num_bags, args.min_bag_size, args.max_bag_size, args.seed,
                                            args.save_path, toggle_proba=args.toggle_proba)

    elif oracle_name == "dial":
        parser.add_argument("--dial_move_proba", type=float, default=0.2)
        args = parser.parse_args()
        DialOracle.generate_dataset(args.num_bags, args.min_bag_size, args.max_bag_size, args.seed, args.save_path,
                                    dial_move_proba=args.dial_move_proba)

    elif oracle_name == "keytreasure":
        print("NOTE: min_bag_size currently unused")
        parser.add_argument("--map_layout", type=str, default="A")
        parser.add_argument("--render", type=int, default=0)
        parser.add_argument("--outcome_ratios", type=list, default=[1/4,1/4,1/2])
        args = parser.parse_args()
        KeyTreasureOracle.generate_dataset(args.num_bags, args.min_bag_size, args.max_bag_size, args.seed, args.save_path,
                                           map_layout=args.map_layout, render=args.render, outcome_ratios=args.outcome_ratios)

    elif oracle_name in {"timertreasure", "movingtreasure", "chargertreasure"}:
        print("NOTE: min_bag_size currently unused")
        if   oracle_name == "timertreasure":   oracle_clz, n_default = TimerTreasureOracle,   10
        elif oracle_name == "movingtreasure":  oracle_clz, n_default = MovingTreasureOracle,  250
        elif oracle_name == "chargertreasure": oracle_clz, n_default = ChargerTreasureOracle, 10
        parser.add_argument("--max_num_per_outcome", type=int, default=n_default)
        parser.add_argument("--render", type=int, default=0)
        parser.add_argument("--plot_outcomes", type=int, default=0)
        parser.add_argument("--wandb", type=int, default=0)
        args = parser.parse_args()
        if args.save_path is None: args.save_path = f"data/oracle/{oracle_clz.name}/{oracle_clz.name}_{args.seed}.csv"
        kwargs = {"render": args.render, "plot_outcomes": args.plot_outcomes, "max_num_per_outcome": args.max_num_per_outcome}
        if args.wandb: import wandb; kwargs["wandb"] = wandb.init(project="mil_datasets", config={"oracle_name": oracle_name, "seed": args.seed})
        oracle_clz.generate_dataset(args.num_bags, args.min_bag_size, args.max_bag_size, args.seed, args.save_path, **kwargs)
    
    elif oracle_name == "lunar_lander":
        parser.add_argument("--plot_outcomes", type=int, default=0)
        args = parser.parse_args()
        kwargs = {
            "load_path": "results/rl_training/lunar_lander/trajectories",
            "plot_outcomes": args.plot_outcomes,
            "label_scale_factor": 0.01 # NOTE:
            }
        LunarLanderOracle.generate_dataset(args.num_bags, None, None, None, args.save_path, **kwargs)

    else:
        raise NotImplementedError('No oracle configured with name {:s}'.format(oracle_name))
