import argparse

import scripts
from interpretability.rl import charger_treasure_interpretability, key_treasure_interpretability, \
    timer_treasure_interpretability, moving_treasure_interpretability
from pytorch_mil.train import get_default_save_path
from pytorch_mil.util import get_device
import matplotlib as mpl

mpl.rcParams.update({"svg.fonttype": "none"})


device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='MIL global interpretability plotting script.')
    scripts.add_dataset_parser_arg(parser, scripts.SYNTHETIC_DATASET_NAMES + scripts.RL_DATASET_NAMES)
    scripts.add_model_clz_parser_arg(parser)
    parser.add_argument('mode', choices=['global', 'probe', 'animate'], help='The dataset to use.')
    parser.add_argument('-q', '--quiet', help='Do not show the plots as they are created.',
                        default=False, action='store_true')
    parser.add_argument('-r', '--repeat_num', help="Repeat number to use. Additional options: 'all', 'best.",
                        default='best')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.mode, args.quiet, args.repeat_num


def run_interpretability_study():
    print('Running global plotting script')
    dataset_name, model_name, mode, quiet, repeat_num = parse_args()
    model_clz = scripts.get_model_clz(dataset_name, model_name)
    csv_path = scripts.get_dataset_path_from_name(dataset_name)
    plotter_clz = get_plotter_clz(dataset_name, mode)

    if repeat_num == 'all':
        repeat_nums = list(range(10))
    elif repeat_num == 'best':
        repeat_nums = [get_best_repeat_num(dataset_name, model_name)]
    else:
        repeat_nums = [int(repeat_num)]

    print('  Dataset: {:s}'.format(dataset_name))
    print('   Path: {:s}'.format(csv_path))
    print('  Model: {:s}'.format(model_name))
    print('   Clz: {:}'.format(model_clz))
    print('  Repeats: {:}'.format(repeat_nums))

    if quiet:
        print("Using quiet mode - plots won't be shown")

    for r in repeat_nums:
        print('-- Plotting repeat {:d} --'.format(r))
        model_path, _, _ = get_default_save_path(dataset_name, model_clz.__name__, repeat=r)
        plotter_clz(device, model_clz, model_path, csv_path, r).run_plotting(show_plots=not quiet)


def get_plotter_clz(dataset_name, mode):
    if dataset_name == 'timer_treasure':
        return timer_treasure_interpretability.get_plotter_clz(mode)
    elif dataset_name == 'moving_treasure':
        return moving_treasure_interpretability.get_plotter_clz(mode)
    elif dataset_name == 'key_treasure':
        return key_treasure_interpretability.get_plotter_clz(mode)
    elif dataset_name == 'charger_treasure':
        return charger_treasure_interpretability.get_plotter_clz(mode)
    raise ValueError('No oracle study registered for dataset {:s}'.format(dataset_name))


def get_best_repeat_num(dataset_name, model_name):
    best_repeats = {
        "timer_treasure": {
            "EmbeddingSpaceLSTM": 2,
            "InstanceSpaceLSTM": 9,
            "CSCInstanceSpaceLSTM": 7,
        },
        "moving_treasure": {
            "EmbeddingSpaceLSTM": 8,
            "InstanceSpaceLSTM": 2,
            "CSCInstanceSpaceLSTM": 4,
        },
        "key_treasure": {
            "EmbeddingSpaceLSTM": 4,
            "InstanceSpaceLSTM": 2,
            "CSCInstanceSpaceLSTM": 5,
        },
        "charger_treasure": {
            "EmbeddingSpaceLSTM": 2,
            "InstanceSpaceLSTM": 0,
            "CSCInstanceSpaceLSTM": 3,
        },
    }
    return best_repeats[dataset_name][model_name]


if __name__ == "__main__":
    run_interpretability_study()
