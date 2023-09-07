"""
Encoder-Decoder architecture: the encoder
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.GRU(
            self.n_features,
            self.hidden_dim,
            batch_first=True,
        )

    def forward(self, X):
        rnn_out, self.hidden = self.basic_rnn(X)

        return rnn_out  # (N, L, F) shape, batch_first


class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.GRU(
            self.n_features,
            self.hidden_dim,
            batch_first=True,
        )
        self.regression = nn.Linear(self.hidden_dim, self.n_features)

    # initializing decoder's hidden state using encoder's final hidden state
    def init_hidden(self, hidden_seq):
        # we only need the final hidden state
        hidden_final = hidden_seq[:, -1:]  # (N, 1, H), batch_first
        # but we need to make it: sequence-first (hidden state must always be)
        self.hidden = hidden_final.permute(1, 0, 2)  # (1, N, H)

    def forward(self, X):  # X is (N, 1, F)
        # the recurrent layer both *uses* and *updates* the hidden state
        batch_first_output, self.hidden = self.basic_rnn(X, self.hidden)
        last_output = batch_first_output[:, -1:]
        out = self.regression(last_output)

        # (N, 1, F), the same shape as the input
        return out.view(-1, 1, self.n_features)
