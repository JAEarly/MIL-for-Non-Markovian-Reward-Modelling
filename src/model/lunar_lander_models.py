import torch

from model import oracle_models
from pytorch_mil.model import aggregator as agg
from pytorch_mil.model import modules as mod


def get_model_clzs():
    return [LLInstanceSpaceNN, LLEmbeddingSpaceLSTM, LLInstanceSpaceLSTM, LLCSCInstanceSpaceLSTM]


def rl_encoder(d_in, dropout):
    return mod.FullyConnectedStack(
        d_in=d_in,
        ds_hid=(128, 64),
        d_out=64,
        dropout=dropout,
        raw_last=False,
    )


ll_ds_hid = (64, 32)
ll_dropout = 0
ll_d_hid = 2
ll_classifier_activation_func = torch.nn.LeakyReLU(1e-6)


class LLInstanceSpaceNN(oracle_models.OracleInstanceSpaceNN):

    def __init__(self, device, d_in, n_expected_dims, dropout=ll_dropout):
        encoder = rl_encoder(d_in, dropout)
        aggregator = agg.InstanceAggregator(
            d_in=encoder.d_out,
            dropout=dropout,
            ds_hid=ll_ds_hid,
            n_classes=1,
            agg_func_name='sum',
            classifier_raw_last=True,
            classifier_activation_func=ll_classifier_activation_func,
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)

    def get_hidden_states(self, bag):
        # This is janky but it just returns zeros as there isn't actually a hidden state in the Instance Space Network
        return torch.zeros((len(bag), self.encoder.d_out))


class LLEmbeddingSpaceLSTM(oracle_models.OracleEmbeddingSpaceLSTM):

    def __init__(self, device, d_in, n_expected_dims, dropout=ll_dropout):
        encoder = rl_encoder(d_in, dropout)
        aggregator = agg.LstmEmbeddingSpaceAggregator(
            d_in=encoder.d_out,
            d_hid=ll_d_hid,
            n_lstm_layers=1,
            bidirectional=False,
            dropout=dropout,
            ds_hid=ll_ds_hid,
            n_classes=1,
            classifier_raw_last=True,
            classifier_activation_func=ll_classifier_activation_func,
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)


class LLInstanceSpaceLSTM(oracle_models.OracleInstanceSpaceLSTM):

    def __init__(self, device, d_in, n_expected_dims, dropout=ll_dropout):
        encoder = rl_encoder(d_in, dropout)
        aggregator = agg.LstmInstanceSpaceAggregator(
            d_in=encoder.d_out,
            d_hid=ll_d_hid,
            n_lstm_layers=1,
            dropout=dropout,
            ds_hid=ll_ds_hid,
            n_classes=1,
            agg_func_name='sum',
            classifier_raw_last=True,
            classifier_activation_func=ll_classifier_activation_func,
        )
        super().__init__(device, 1, n_expected_dims, encoder, aggregator)


class LLCSCInstanceSpaceLSTM(oracle_models.OracleCSCInstanceSpaceLSTM):

    def __init__(self, device, d_in, n_expected_dims, dropout=ll_dropout):
        encoder = rl_encoder(d_in, dropout)
        aggregator = agg.LstmCSCInstanceSpaceAggregator(
            d_in=encoder.d_out,
            d_hid=ll_d_hid,
            n_lstm_layers=1,
            dropout=dropout,
            ds_hid=ll_ds_hid,
            n_classes=1,
            agg_func_name='sum',
            classifier_raw_last=True,
            classifier_activation_func=ll_classifier_activation_func,
        )

        super().__init__(device, 1, n_expected_dims, encoder, aggregator)
