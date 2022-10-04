from abc import ABC, abstractmethod

import torch

from pytorch_mil.model import models


def returns_and_rewards_from_instance_predictions(bag_prediction, instance_predictions):
    # Check bag prediction is the sum of instance interpretations as per model definition
    assert torch.sum(instance_predictions, dim=0) == bag_prediction
    # Generate returns throughout the episode by calculating the cumulative sum of the rewards
    # Check last return value is (almost) equal to the bag prediction
    rewards = instance_predictions
    returns = torch.cumsum(rewards, dim=0)
    assert abs(returns[-1] - bag_prediction) < 0.0001
    return returns, rewards


def returns_and_rewards_from_cumulative_predictions(bag_prediction, cumulative_predictions):
    # Reward for the current timestep is the different between the returns for the current and previous timesteps
    rewards = torch.zeros_like(cumulative_predictions)
    for i in range(0, len(rewards)):
        rewards[i] = cumulative_predictions[i] if i == 0 else cumulative_predictions[i] - cumulative_predictions[i - 1]
    # Generate returns throughout the episode by calculating the cumulative sum of the rewards
    # Check last return value is (almost) equal to the bag prediction
    returns = torch.cumsum(rewards, dim=0)
    assert abs(returns[-1] - bag_prediction) < 0.0001
    return returns, rewards


class OracleMILModelMixin(ABC):

    @abstractmethod
    def forward_returns_and_rewards(self, model_input):
        pass


class OracleInstanceSpaceNN(OracleMILModelMixin, models.InstanceSpaceNN, ABC):

    def forward_returns_and_rewards(self, model_input):
        if len(model_input.shape) != 2:  # n_instance x n_features
            raise ValueError('Invalid input to model, shape is: {:}'.format(model_input.shape))
        bag_prediction, instance_predictions = self.forward_verbose(model_input)
        return returns_and_rewards_from_instance_predictions(bag_prediction, instance_predictions.squeeze())


class OracleEmbeddingSpaceLSTM(OracleMILModelMixin, models.EmbeddingSpaceLSTM, ABC):

    def forward_returns_and_rewards(self, model_input):
        if len(model_input.shape) != 2:  # n_instance x n_features
            raise ValueError('Invalid input to model, shape is: {:}'.format(model_input.shape))
        bag_prediction, cumulative_predictions = self.forward_verbose(model_input)
        return returns_and_rewards_from_cumulative_predictions(bag_prediction, cumulative_predictions.squeeze())


class OracleInstanceSpaceLSTM(OracleMILModelMixin, models.InstanceSpaceLSTM, ABC):

    def forward_returns_and_rewards(self, model_input):
        if len(model_input.shape) != 2:  # n_instance x n_features
            raise ValueError('Invalid input to model, shape is: {:}'.format(model_input.shape))
        bag_prediction, instance_predictions = self.forward_verbose(model_input)
        return returns_and_rewards_from_instance_predictions(bag_prediction, instance_predictions.squeeze())


class OracleCSCInstanceSpaceLSTM(OracleMILModelMixin, models.CSCInstanceSpaceLSTM, ABC):

    def forward_returns_and_rewards(self, model_input):
        if len(model_input.shape) != 2:  # n_instance x n_features
            raise ValueError('Invalid input to model, shape is: {:}'.format(model_input.shape))
        bag_prediction, instance_predictions = self.forward_verbose(model_input)
        return returns_and_rewards_from_instance_predictions(bag_prediction, instance_predictions.squeeze())
