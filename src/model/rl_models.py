import torch

from model import oracle_models
from pytorch_mil.model import aggregator as agg
from pytorch_mil.model import modules as mod


def get_model_clzs():
    return [RLInstanceSpaceNN, RLEmbeddingSpaceLSTM, RLInstanceSpaceLSTM, RLCSCInstanceSpaceLSTM]


def rl_encoder(d_in, dropout):
    return mod.FullyConnectedStack(
        d_in=d_in,
        ds_hid=(64, 32),
        d_out=32,
        dropout=dropout,
        raw_last=False,
    )


class RLInstanceSpaceNN(oracle_models.OracleInstanceSpaceNN):

    def __init__(self, device, d_in, n_expected_dims, dropout=0.1):
        encoder = rl_encoder(d_in, dropout)
        aggregator = agg.InstanceAggregator(
            d_in=encoder.d_out,
            dropout=dropout,
            ds_hid=(32, 16),
            n_classes=1,
            agg_func_name='sum',
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)

    def get_hidden_states(self, bag):
        # This is janky but it just returns zeros as there isn't actually a hidden state in the Instance Space Network
        return torch.zeros((len(bag), self.encoder.d_out))


class RLEmbeddingSpaceLSTM(oracle_models.OracleEmbeddingSpaceLSTM):

    def __init__(self, device, d_in, n_expected_dims, dropout=0.1):
        encoder = rl_encoder(d_in, dropout)
        aggregator = agg.LstmEmbeddingSpaceAggregator(
            d_in=encoder.d_out,
            d_hid=2,
            n_lstm_layers=1,
            bidirectional=False,
            dropout=dropout,
            ds_hid=(32, 16),
            n_classes=1,
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)


class RLInstanceSpaceLSTM(oracle_models.OracleInstanceSpaceLSTM):

    def __init__(self, device, d_in, n_expected_dims, dropout=0.1):
        encoder = rl_encoder(d_in, dropout)
        aggregator = agg.LstmInstanceSpaceAggregator(
            d_in=encoder.d_out,
            d_hid=2,
            n_lstm_layers=1,
            dropout=dropout,
            ds_hid=(32, 16),
            n_classes=1,
            agg_func_name='sum',
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)


class RLCSCInstanceSpaceLSTM(oracle_models.OracleCSCInstanceSpaceLSTM):

    def __init__(self, device, d_in, n_expected_dims, dropout=0.1):
        encoder = rl_encoder(d_in, dropout)
        aggregator = agg.LstmCSCInstanceSpaceAggregator(
            d_in=encoder.d_out,
            d_hid=2,
            n_lstm_layers=1,
            dropout=dropout,
            ds_hid=(32, 16),
            n_classes=1,
            agg_func_name='sum',
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)
