import yaml

from .rl.chargertreasure_dataset import ChargerTreasureDataset
from .rl.keytreasure_dataset import KeyTreasureDataset
from .rl.lunar_lander_dataset import LunarLanderDataset
from .rl.movingtreasure_dataset import MovingTreasureDataset
from .rl.timertreasure_dataset import TimerTreasureDataset
from .synthetic.dial_dataset import DialOracleDataset
from .synthetic.push_switch_dataset import PushSwitchOracleDataset
from .synthetic.toggle_switch_dataset import ToggleSwitchOracleDataset

# Populated in the parse_yaml function
DATASET_PATHS = {}


def parse_yaml():
    if len(DATASET_PATHS) == 0:
        stream = open("data/datasets.yaml", 'r')
        for dataset_type, dataset_details in yaml.safe_load(stream).items():
            for name, paths in dataset_details.items():
                if name in DATASET_PATHS.keys():
                    raise ValueError('Duplicate dataset name when parsing yaml: {:s}'.format(name))
                DATASET_PATHS[name] = paths


parse_yaml()


def get_dataset_path_from_name(dataset_name):
    if dataset_name not in DATASET_PATHS:
        raise NotImplementedError('No dataset found for name {:s}'.format(dataset_name))
    dataset_paths = DATASET_PATHS[dataset_name]
    if len(dataset_paths) > 1:
        print('More than one dataset file available for dataset {:s}. Using first by default.'.format(dataset_name))
    return dataset_paths[0]
