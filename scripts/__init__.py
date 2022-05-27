import inspect

import yaml

import interpretability
from model import synthetic_models, rl_models, training

SYNTHETIC_DATASET_NAMES = ['toggle_switch', 'push_switch', 'dial']
RL_DATASET_NAMES = ['key_treasure', 'moving_treasure', 'timer_treasure', 'charger_treasure']

MODEL_NAMES = ['InstanceSpaceNN', 'EmbeddingSpaceLSTM', 'InstanceSpaceLSTM', 'CSCInstanceSpaceLSTM']

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


def add_dataset_parser_arg(parser, dataset_names):
    parser.add_argument('dataset_name', choices=dataset_names, help='The dataset to use.')


def add_model_clz_parser_arg(parser):
    parser.add_argument('model_name', choices=MODEL_NAMES, help='The model to use.')


def get_model_clz(dataset_name, model_name):
    if dataset_name in SYNTHETIC_DATASET_NAMES:
        model_clzs = synthetic_models.get_model_clzs()
    elif dataset_name in RL_DATASET_NAMES:
        model_clzs = rl_models.get_model_clzs()
    else:
        raise ValueError('No models registered for dataset {:s}'.format(dataset_name))
    for model_clz in model_clzs:
        if model_name in [base_clz.__name__ for base_clz in inspect.getmro(model_clz)]:
            return model_clz
    raise ValueError('No model with name {:s} found for dataset {:s}'.format(model_name, dataset_name))


def get_dataset_path_from_name(dataset_name):
    if dataset_name not in DATASET_PATHS:
        raise NotImplementedError('No dataset found for name {:s}'.format(dataset_name))
    dataset_paths = DATASET_PATHS[dataset_name]
    if len(dataset_paths) > 1:
        print('More than one dataset file available for dataset {:s}. Using first by default.'.format(dataset_name))
    return dataset_paths[0]


def get_trainer_clz(dataset_name):
    if dataset_name == 'toggle_switch':
        return training.ToggleSwitchOracleTrainer
    elif dataset_name == 'push_switch':
        return training.PushSwitchOracleTrainer
    elif dataset_name == 'dial':
        return training.DialOracleTrainer
    elif dataset_name == 'key_treasure':
        return training.KeyTreasureOracleTrainer
    elif dataset_name == 'moving_treasure':
        return training.MovingTreasureOracleTrainer
    elif dataset_name == 'timer_treasure':
        return training.TimerTreasureOracleTrainer
    elif dataset_name == 'charger_treasure':
        return training.ChargerTreasureOracleTrainer
    raise ValueError('No oracle trainer registered for dataset {:s}'.format(dataset_name))
