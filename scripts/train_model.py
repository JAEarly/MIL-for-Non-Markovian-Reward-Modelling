import argparse

import dataset
import scripts
from pytorch_mil.train import DEFAULT_SEEDS
from pytorch_mil.util import get_device

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='MIL Oracle Datasets training script.')
    scripts.add_dataset_parser_arg(parser, scripts.SYNTHETIC_DATASET_NAMES
                                   + scripts.RL_DATASET_NAMES
                                   + scripts.LL_DATASET_NAMES)
    scripts.add_model_clz_parser_arg(parser)
    parser.add_argument('-s', '--seeds', default=",".join(str(s) for s in DEFAULT_SEEDS), type=str,
                        help='The seeds for the training. Should be at least as long as the number of repeats.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to train (>=1).')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.seeds, args.n_repeats


def run_training():
    dataset_name, model_name, seeds, n_repeats = parse_args()

    # Parse seed list
    seeds = [int(s) for s in seeds.split(",")]
    if len(seeds) < n_repeats:
        raise ValueError('Not enough seeds provided for {:d} repeats'.format(n_repeats))
    seeds = seeds[:n_repeats]

    model_clz = scripts.get_model_clz(dataset_name, model_name)
    trainer_clz = scripts.get_trainer_clz(dataset_name)
    csv_path = dataset.get_dataset_path_from_name(dataset_name)
    trainer = trainer_clz(device, model_clz, dataset_name, csv_path)

    print('Starting {:s} training'.format(dataset_name))
    print('  Using device {:}'.format(device))
    print('  Using model {:}'.format(model_clz))
    print('  Using trainer {:}'.format(trainer_clz))
    print('  Loading data from {:s}'.format(csv_path))
    print('  Seeds: {:}'.format(seeds))

    if n_repeats > 1:
        print('  Training using multiple trainer')
        trainer.train_multiple(n_repeats=n_repeats, seeds=seeds)
    elif n_repeats == 1:
        print('  Training using single trainer')
        trainer.train_single(seed=seeds[0])
    else:
        raise ValueError("Invalid number of repeats for training: {:d}".format(n_repeats))


if __name__ == "__main__":
    run_training()
