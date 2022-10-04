import inspect

from model import synthetic_models, rl_models, lunar_lander_models, training

SYNTHETIC_DATASET_NAMES = ['toggle_switch', 'push_switch', 'dial']
RL_DATASET_NAMES = ['key_treasure', 'moving_treasure', 'timer_treasure', 'charger_treasure']
LL_DATASET_NAMES = ['lunar_lander']

MODEL_NAMES = ['InstanceSpaceNN', 'EmbeddingSpaceLSTM', 'InstanceSpaceLSTM', 'CSCInstanceSpaceLSTM']


def add_dataset_parser_arg(parser, dataset_names):
    parser.add_argument('dataset_name', choices=dataset_names, help='The dataset to use.')


def add_model_clz_parser_arg(parser):
    parser.add_argument('model_name', choices=MODEL_NAMES, help='The model to use.')


def get_model_clz(dataset_name, model_name):
    if dataset_name in SYNTHETIC_DATASET_NAMES:
        model_clzs = synthetic_models.get_model_clzs()
    elif dataset_name in RL_DATASET_NAMES:
        model_clzs = rl_models.get_model_clzs()
    elif dataset_name in LL_DATASET_NAMES:
        model_clzs = lunar_lander_models.get_model_clzs()
    else:
        raise ValueError('No models registered for dataset {:s}'.format(dataset_name))
    for model_clz in model_clzs:
        if model_name in [base_clz.__name__ for base_clz in inspect.getmro(model_clz)]:
            return model_clz
    raise ValueError('No model with name {:s} found for dataset {:s}'.format(model_name, dataset_name))


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
    elif dataset_name == 'lunar_lander':
        return training.LunarLanderOracleTrainer
    raise ValueError('No oracle trainer registered for dataset {:s}'.format(dataset_name))
