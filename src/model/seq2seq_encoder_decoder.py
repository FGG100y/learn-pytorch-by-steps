"""
Encoder-Decoder architecture: the encoder
"""
import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, input_len, target_len,
                 teacher_forcing_proba=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_len = input_len
        self.target_len = target_len
        self.teacher_forcing_proba = teacher_forcing_proba
        self.outputs = None

    def init_outputs(self, batch_size):
        device = next(self.parameters()).device
        # (N, L (target), F)
        self.outputs = torch.zeros(batch_size,
                                   self.target_len,
                                   self.encoder.n_features).to(device)

    def store_output(self, i, out):
        self.outputs[:, i:i+1, :] = out

    def forward(self, X):
        # splits the data in source and target sequences
        # the target seq will be empty in testing mode
        # (N, L, F)
        source_seq = X[:, :self.input_len, :]
        target_seq = X[:, self.input_len:, :]
        self.init_outputs(X.shape[0])

        # Encoder expected (N, L, F)
        hidden_seq = self.encoder(source_seq)
        # Output is (N, L, H)
        self.decoder.init_hidden(hidden_seq)

        # the last input of the encoder is also the first input of the decoder
        dec_inputs = source_seq[:, -1:, :]

        # generates as many outputs as the target length
        for i in range(self.target_len):
            # output of decoder is (N, 1, F)
            out = self.decoder(dec_inputs)
            self.store_output(i, out)

            proba = self.teacher_forcing_proba
            # in evaluation / testing the target sequence is unknown,
            # so we cannot use teacher forcing
            if not self.training:
                proba = 0

            # if it is teacher forcing
            if torch.rand(1) <= proba:
                # takes the actual element
                dec_inputs = target_seq[:, i:i+1, :]
            else:
                # otherwise uses the last prediction
                dec_inputs = out

        return self.outputs


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
