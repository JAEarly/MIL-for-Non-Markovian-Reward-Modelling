from abc import ABC

from overrides import overrides

import dataset
from pytorch_mil.train import get_default_save_path
from pytorch_mil.train.train_base import MinimiseRegressionTrainer, NetTrainerMixin


class RLMILTrainer(MinimiseRegressionTrainer, ABC):

    @overrides
    def create_model(self):
        if self.model_params is not None:
            return self.model_clz(self.device, self.dataset_clz.d_in, self.dataset_clz.n_expected_dims,
                                  **self.model_params)
        return self.model_clz(self.device, self.dataset_clz.d_in, self.dataset_clz.n_expected_dims)


class ToggleSwitchOracleTrainer(NetTrainerMixin, RLMILTrainer):

    dataset_clz = dataset.ToggleSwitchOracleDataset

    def __init__(self, device, model_clz, dataset_name, csv_path, model_params=None, train_params_override=None):
        dataset_params = {
            'csv_path': csv_path,
        }
        super().__init__(device, model_clz, dataset_name, dataset_params=dataset_params,
                         model_params=model_params, train_params_override=train_params_override)

    def get_default_train_params(self):
        return {
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'patience': 20,
            'n_epochs': 100,
        }


class PushSwitchOracleTrainer(NetTrainerMixin, RLMILTrainer):

    dataset_clz = dataset.PushSwitchOracleDataset

    def __init__(self, device, model_clz, dataset_name, csv_path, model_params=None, train_params_override=None):
        dataset_params = {
            'csv_path': csv_path,
        }
        super().__init__(device, model_clz, dataset_name, dataset_params=dataset_params,
                         model_params=model_params, train_params_override=train_params_override)

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 0,
            'patience': 30,
            'n_epochs': 150,
        }


class DialOracleTrainer(NetTrainerMixin, RLMILTrainer):

    dataset_clz = dataset.DialOracleDataset

    def __init__(self, device, model_clz, dataset_name, csv_path, model_params=None, train_params_override=None):
        dataset_params = {
            'csv_path': csv_path,
        }
        super().__init__(device, model_clz, dataset_name, dataset_params=dataset_params,
                         model_params=model_params, train_params_override=train_params_override)

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 0,
            'patience': 30,
            'n_epochs': 150,
        }


class KeyTreasureOracleTrainer(NetTrainerMixin, RLMILTrainer):

    dataset_clz = dataset.KeyTreasureDataset

    def __init__(self, device, model_clz, dataset_name, csv_path, model_params=None, train_params_override=None):
        dataset_params = {
            'csv_path': csv_path,
        }
        super().__init__(device, model_clz, dataset_name, dataset_params=dataset_params,
                         model_params=model_params, train_params_override=train_params_override)

    def get_default_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 0,
            'patience': 30,
            'n_epochs': 150,
        }


class MovingTreasureOracleTrainer(NetTrainerMixin, RLMILTrainer):

    dataset_clz = dataset.MovingTreasureDataset

    def __init__(self, device, model_clz, dataset_name, csv_path, model_params=None, train_params_override=None):
        dataset_params = {
            'csv_path': csv_path,
        }
        super().__init__(device, model_clz, dataset_name, dataset_params=dataset_params,
                         model_params=model_params, train_params_override=train_params_override)

    def get_default_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 0,
            'patience': 50,
            'n_epochs': 250,
        }


class TimerTreasureOracleTrainer(NetTrainerMixin, RLMILTrainer):

    dataset_clz = dataset.TimerTreasureDataset

    def __init__(self, device, model_clz, dataset_name, csv_path, model_params=None, train_params_override=None):
        dataset_params = {
            'csv_path': csv_path,
        }
        super().__init__(device, model_clz, dataset_name, dataset_params=dataset_params,
                         model_params=model_params, train_params_override=train_params_override)

    def get_default_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 0,
            'patience': 50,
            'n_epochs': 250,
        }


class ChargerTreasureOracleTrainer(NetTrainerMixin, RLMILTrainer):

    dataset_clz = dataset.ChargerTreasureDataset

    def __init__(self, device, model_clz, dataset_name, csv_path, model_params=None, train_params_override=None):
        dataset_params = {
            'csv_path': csv_path,
        }
        super().__init__(device, model_clz, dataset_name, dataset_params=dataset_params,
                         model_params=model_params, train_params_override=train_params_override)

    def get_default_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 0,
            'patience': 50,
            'n_epochs': 250,
        }


class LunarLanderOracleTrainer(NetTrainerMixin, RLMILTrainer):

    dataset_clz = dataset.LunarLanderDataset

    def __init__(self, device, model_clz, dataset_name, csv_path, model_params=None, train_params_override=None):
        dataset_params = {
            'csv_path': csv_path,
        }
        super().__init__(device, model_clz, dataset_name, dataset_params=dataset_params,
                         model_params=model_params, train_params_override=train_params_override)

    def get_default_train_params(self):
        return {
            'lr': 1e-4,
            'weight_decay': 0,
            'patience': 30,
            'n_epochs': 200,
        }

    @overrides
    def train_single(self, seed=5, save_model=True, show_plot=True, verbose=True, trial=None):
        self.seed = seed
        super().train_single(seed=seed, save_model=save_model, show_plot=show_plot, verbose=verbose, trial=trial)

    @overrides
    def get_model_save_path(self, model, repeat):
        path, save_dir, _ = get_default_save_path(self.dataset_name, self.model_name,
                                                  param_save_string=model.get_param_save_string(), repeat=repeat)
        path += "." + str(self.seed)
        return path, save_dir
