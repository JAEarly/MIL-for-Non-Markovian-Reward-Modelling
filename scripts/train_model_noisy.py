import argparse

import dataset
import scripts
from pytorch_mil.train import DEFAULT_SEEDS
from pytorch_mil.util import get_device

device = get_device()

NOISE_SEEDS = [24, 35, 62]


def parse_args():
    parser = argparse.ArgumentParser(description='MIL Oracle Datasets training script.')
    scripts.add_dataset_parser_arg(parser, scripts.SYNTHETIC_DATASET_NAMES + scripts.RL_DATASET_NAMES)
    scripts.add_model_clz_parser_arg(parser)
    parser.add_argument('-d', '--dataset_seeds', default=",".join(str(s) for s in DEFAULT_SEEDS), type=str,
                        help='The seeds for the dataset split. Should be at least as long as the number of repeats.')
    parser.add_argument('-n', '--noise_seeds', default=",".join(str(s) for s in NOISE_SEEDS), type=str,
                        help='The seeds for the dataset noise. Should be at least as long as the number of repeats.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to train (>=1).')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.dataset_seeds, args.noise_seeds, args.n_repeats


def run_training():
    dataset_name, model_name, dataset_seeds, noise_seeds, n_repeats = parse_args()

    # Parse dataset seed list
    dataset_seeds = [int(s) for s in dataset_seeds.split(",")]
    if len(dataset_seeds) < n_repeats:
        raise ValueError('Not enough seeds provided for {:d} repeats'.format(n_repeats))
    dataset_seeds = dataset_seeds[:n_repeats]

    # Parse noise seed list
    noise_seeds = [int(s) for s in noise_seeds.split(",")]
    if len(noise_seeds) < n_repeats:
        raise ValueError('Not enough seeds provided for {:d} repeats'.format(n_repeats))
    noise_seeds = noise_seeds[:n_repeats]

    model_clz = scripts.get_model_clz(dataset_name, model_name)
    trainer_clz = scripts.get_trainer_clz(dataset_name)
    csv_path = dataset.get_dataset_path_from_name(dataset_name)

    noise_levels = [0.05, 0.15, 0.25, 0.35, 0.45]

    print('Starting noisy {:s} training'.format(dataset_name))
    print('  Using device {:}'.format(device))
    print('  Using model {:}'.format(model_clz))
    print('  Using trainer {:}'.format(trainer_clz))
    print('  Loading data from {:s}'.format(csv_path))
    print('  Dataset Seeds: {:}'.format(dataset_seeds))
    print('  Noise Seeds: {:}'.format(noise_seeds))

    # Run for each noise seed
    for run_idx, noise_seed in enumerate(noise_seeds):
        dataset_seed = dataset_seeds[run_idx]
        print('Starting run {:d}'.format(run_idx + 1))
        print('  Noise Seed: {:d}'.format(noise_seed))
        print('  Dataset Seed: {:d}'.format(dataset_seed))
        # Run for each noise level
        for noise_level in noise_levels:
            print('  Running for noise level {:.2f}'.format(noise_level))

            # Create custom trainer class with correct noise level and seeding
            class TrainerWithNoise(trainer_clz):

                # Override load datasets to inject target noise
                def load_datasets(self, seed=None):
                    train_dataset, val_dataset, test_dataset = super().load_datasets(seed)
                    train_dataset.permute_targets(noise_level, noise_seed, verbose=False)
                    return train_dataset, val_dataset, test_dataset

                def get_model_save_path(self, model, repeat):
                    return get_noisy_model_path(self.dataset_name, model_clz.__name__, noise_level, run_idx)

            trainer = TrainerWithNoise(device, model_clz, dataset_name, csv_path)
            trainer.train_single(seed=dataset_seed, show_plot=False, verbose=False)


def get_noisy_model_path(dataset_name, model_name, noise_level, run_idx):
    save_dir = "models/{:s}/noisy/{:s}_{:s}".format(dataset_name, dataset_name, model_name)
    file_name = "{:s}_{:s}_{:.2f}_{:d}.pkl".format(dataset_name, model_name, noise_level, run_idx)
    path = "{:s}/{:s}".format(save_dir, file_name)
    return path, save_dir


if __name__ == "__main__":
    run_training()
