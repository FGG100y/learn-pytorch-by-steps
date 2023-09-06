import torch
import torch.nn as nn
from torch.nn.utils import rnn as rnn_utils


class SquareModelPacked(nn.Module):
    # assuming bidirectional LSTM and expected packed sequences as input
    def __init__(self, n_features, hidden_dim, n_outputs):
        super(SquareModelPacked, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = None
        self.cell = None
        # simple LSTM:
        self.basic_rnn = nn.LSTM(
            self.n_features,
            self.hidden_dim,
            bidirectional=True,
        )
        # classifer
        self.classifier = nn.Linear(2 * self.hidden_dim, self.n_outputs)

    def forward(self, X):
        # X is PACKED sequence
        # final hidden state is (2, N, H)
        # final cell state is (2, N, H)
        birnn_out, (self.hidden, self.cell) = self.basic_rnn(X)
        # unpack the output (N, L, 2*H)
        batch_first_output, seq_sizes = rnn_utils.pad_packed_sequence(
            birnn_out, batch_first=True
        )
        # only last item in sequence (N, 1, 2*H)
        # classifer will output (N, 1, n_outputs)
        seq_idx = torch.arange(seq_sizes.size(0))
        last_output = batch_first_output[seq_idx, seq_sizes - 1]
        out = self.classifier(last_output)

        # final output is (N, n_outputs)
        return out.view(-1, self.n_outputs)
