import torch

from model import oracle_models
from pytorch_mil.model import aggregator as agg
from pytorch_mil.model import modules as mod


def get_model_clzs():
    return [SyntheticInstanceSpaceNN, SyntheticEmbeddingSpaceLSTM,
            SyntheticInstanceSpaceLSTM, SyntheticCSCInstanceSpaceLSTM]


def synthetic_encoder(d_in):
    return mod.FullyConnectedStack(
        d_in=d_in,
        ds_hid=(),
        d_out=d_in,
        dropout=0,
        raw_last=False,
    )


class SyntheticInstanceSpaceNN(oracle_models.OracleInstanceSpaceNN):

    def __init__(self, device, d_in, n_expected_dims):
        encoder = synthetic_encoder(d_in)
        aggregator = agg.InstanceAggregator(
            d_in=encoder.d_out,
            ds_hid=(),
            n_classes=1,
            dropout=0,
            agg_func_name='sum',
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)

    def get_hidden_states(self, bag):
        # This is janky but it just returns zeros as there isn't actually a hidden state in the Instance Space Network
        return torch.zeros((len(bag), self.encoder.d_out))


class SyntheticEmbeddingSpaceLSTM(oracle_models.OracleEmbeddingSpaceLSTM):

    def __init__(self, device, d_in, n_expected_dims):
        encoder = synthetic_encoder(d_in)
        aggregator = agg.LstmEmbeddingSpaceAggregator(
            d_in=encoder.d_out,
            d_hid=2,
            n_lstm_layers=1,
            bidirectional=False,
            dropout=0,
            ds_hid=(),
            n_classes=1,
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)


class SyntheticInstanceSpaceLSTM(oracle_models.OracleInstanceSpaceLSTM):

    def __init__(self, device, d_in, n_expected_dims):
        encoder = synthetic_encoder(d_in)
        aggregator = agg.LstmInstanceSpaceAggregator(
            d_in=encoder.d_out,
            d_hid=2,
            n_lstm_layers=1,
            dropout=0,
            ds_hid=(),
            n_classes=1,
            agg_func_name='sum',
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)


class SyntheticCSCInstanceSpaceLSTM(oracle_models.OracleCSCInstanceSpaceLSTM):

    def __init__(self, device, d_in, n_expected_dims):
        encoder = synthetic_encoder(d_in)
        aggregator = agg.LstmCSCInstanceSpaceAggregator(
            d_in=encoder.d_out,
            d_hid=2,
            n_lstm_layers=1,
            dropout=0,
            ds_hid=(),
            n_classes=1,
            agg_func_name='sum',
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)
