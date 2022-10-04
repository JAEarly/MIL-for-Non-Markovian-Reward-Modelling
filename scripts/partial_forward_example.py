from random import shuffle
from time import sleep

import torch
from tqdm import tqdm

import dataset as rl_datasets
from model import lunar_lander_models
from pytorch_mil.train import get_default_save_path
from pytorch_mil.util import get_device

device = get_device()


def run():
    print('Starting partial forward example')

    datasets = {
        # "charger_treasure": rl_datasets.ChargerTreasureDataset,
        # "key_treasure": rl_datasets.KeyTreasureDataset,
        # "moving_treasure": rl_datasets.MovingTreasureDataset,
        # "timer_treasure": rl_datasets.TimerTreasureDataset,
        "lunar_lander": rl_datasets.LunarLanderDataset,
    }

    models = [
        lunar_lander_models.LLEmbeddingSpaceLSTM,
        lunar_lander_models.LLInstanceSpaceLSTM,
        lunar_lander_models.LLCSCInstanceSpaceLSTM
    ]

    n_tests = len(datasets) * len(models)
    test_idx = 0
    for dataset_name, dataset_clz in datasets.items():
        for model_clz in models:
            print('\nRunning test {:d}/{:d} - {:s} {:s}'.format(test_idx + 1, n_tests, dataset_name, model_clz.__name__))
            run_single(dataset_name, dataset_clz, model_clz)
            # Place nicely with tqdm
            sleep(1)
            print('Passed!')
            test_idx += 1
    print('------')
    print('All Passed!')


def run_single(dataset_name, dataset_clz, model_clz):
    # Actually initialise stuff
    print('Loading dataset...')
    csv_path = rl_datasets.get_dataset_path_from_name(dataset_name)
    print(' Using data from {:s}'.format(csv_path))
    dataset = dataset_clz.create_complete_dataset(csv_path=csv_path)
    print(' Done')
    print('Loading model...')
    model_path, _, _ = get_default_save_path(dataset_name, model_clz.__name__, repeat=0)
    model = model_clz.load_model(device, model_path, dataset_clz.d_in, dataset_clz.n_expected_dims)

    print(' Using model file {:s}'.format(model_path))
    model.eval()
    print(' Done')

    # Clone model init hidden states
    init_hidden = model.aggregator.lstm_block.init_hidden
    init_cell = model.aggregator.lstm_block.init_cell
    print('-- Init Hidden --')
    print(init_hidden)
    print('-- Init Cell --')
    print(init_cell)

    # Play nicely with tqdm
    sleep(1)

    # Loop through dataset to verify the running predictions match the normal predictions
    #  Shuffle dataset order to get a good spread in our verification
    #  Then only select a subset to test
    idxs = list(range(len(dataset)))
    shuffle(idxs)
    n_to_test_on = 250
    idxs = idxs[:n_to_test_on]
    for idx in tqdm(idxs, desc='Verifying partial forward', leave=False):
        # Load from dataset
        bag, bag_label, instance_labels = dataset[idx]

        # Get the original predictions by passing the complete bag through the model
        orig_return_preds, orig_reward_preds = model.forward_returns_and_rewards(bag)
        orig_return_pred = orig_return_preds[-1]

        # Initialise our 'running' states with the initial state
        running_hidden = init_hidden
        running_cell = init_cell
        running_instance_preds = torch.zeros(len(bag))
        prev_cumulative_bag_prediction = torch.as_tensor([0], dtype=torch.float32).to(device)
        # Iterate through instances in the bag
        for instance_idx, instance in enumerate(bag):
            # Pass instance and running states through model
            out = model.partial_forward(instance, running_hidden, running_cell, prev_cumulative_bag_prediction)
            # Get instance prediction and update running states
            instance_prediction, running_hidden, running_cell = out
            running_instance_preds[instance_idx] = instance_prediction
            prev_cumulative_bag_prediction += instance_prediction.squeeze()

        # Create bag prediction by aggregation running instance preds
        running_bag_pred = torch.sum(running_instance_preds)

        # Check bag predictions are similar (enough)
        bag_diff = (orig_return_pred - running_bag_pred).item()
        if abs(bag_diff) > 1e-3:
            print('Bag diff too high!', bag_diff)
            print('Original:', orig_return_pred)
            print(' Running:', running_bag_pred)
            raise ValueError('Test Failed!')

        # Check all instance predictions are similar (enough)
        instance_diff = (orig_reward_preds.detach().cpu().squeeze() - running_instance_preds)
        for i in range(len(orig_reward_preds)):
            if abs(instance_diff[i]) > 1e-4:
                print('Instance diff too high!', instance_diff[i])
                print('Original:', orig_reward_preds[i])
                print(' Running:', running_instance_preds[i])
                raise ValueError('Test Failed!')


if __name__ == "__main__":
    run()
