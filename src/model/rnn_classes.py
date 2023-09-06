import torch
import torch.nn as nn
from torch.nn.utils import rnn as rnn_utils


class SquareModelOne(nn.Module):
    # handle all types of recurrent layers (RNN, GRU, or LSTM),
    # and all types of sequences (packed or not)
    def __init__(self, n_features, hidden_dim, n_outputs, rnn_layer=nn.LSTM,
                 **kwargs):
        super(SquareModelOne, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = None
        self.cell = None
        # simple LSTM:
        self.basic_rnn = rnn_layer(
            self.n_features,
            self.hidden_dim,
            batch_first=True,
            **kwargs,
        )
        output_dim = (self.basic_rnn.bidirectional + 1) * self.hidden_dim
        # classifer
        self.classifier = nn.Linear(output_dim, self.n_outputs)

    def forward(self, X):
        is_packed = isinstance(X, nn.utils.rnn.PackedSequence)
        # X is PACKED sequence, there is no need to permute

        rnn_out, self.hidden = self.basic_rnn(X)
        if isinstance(self.basic_rnn, nn.LSTM):
            self.hidden, self.cell = self.hidden

        if is_packed:
            # unpack the output (N, L, 2*H)
            batch_first_output, seq_sizes = rnn_utils.pad_packed_sequence(
                rnn_out, batch_first=True
            )
            seq_slice = torch.arange(seq_sizes.size(0))
        else:
            batch_first_output = rnn_out
            seq_sizes = 0   # so that it is -1 as the last output
            seq_slice = slice(None, None, None)  # same as ":"

        # only last item in sequence (N, 1, H)
        last_output = batch_first_output[seq_slice, seq_sizes - 1]

        # classifer will output (N, 1, n_outputs)
        out = self.classifier(last_output)

        # final output is (N, n_outputs)
        return out.view(-1, self.n_outputs)
