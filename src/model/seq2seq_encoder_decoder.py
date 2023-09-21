"""
Encoder-Decoder architecture: the encoder
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DecoderAttn(nn.Module):
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
        self.attn = Attention(self.hidden_dim)
        self.regression = nn.Linear(2*self.hidden_dim, self.n_features)

    # initializing decoder's hidden state using encoder's final hidden state
    def init_hidden(self, hidden_seq):
        # we only need the final hidden state
        hidden_final = hidden_seq[:, -1:]  # (N, 1, H), batch_first
        # but we need to make it: sequence-first (hidden state must always be)
        self.hidden = hidden_final.permute(1, 0, 2)  # (1, N, H)

    def forward(self, X, mask=None):  # X is (N, 1, F)
        # the recurrent layer both *uses* and *updates* the hidden state
        batch_first_output, self.hidden = self.basic_rnn(X, self.hidden)
        query = batch_first_output[:, -1:]
        # attention:
        context = self.attn(query, mask=mask)
        concatenated = torch.cat([context, query], axis=-1)
        out = self.regression(concatenated)

        # (N, 1, F), the same shape as the input
        return out.view(-1, 1, self.n_features)


class Attention(nn.Module):
    def __init__(self, hidden_dim, input_dim=None, proj_values=False):
        super().__init__()
        self.d_k = hidden_dim
        self.input_dim = hidden_dim if input_dim is None else input_dim
        self.proj_values = proj_values
        # affine transformations for Q,K,V
        self.linear_query = nn.Linear(self.input_dim, hidden_dim)
        self.linear_key = nn.Linear(self.input_dim, hidden_dim)
        self.linear_values = nn.Linear(self.input_dim, hidden_dim)
        self.alphas = None

    # receive a batch-first hidden states from encoder
    def init_keys(self, keys):
        self.keys = keys
        self.proj_keys = self.linear_key(self.keys)
        self.values = self.linear_query(self.keys) if self.proj_values else self.keys

    def score_function(self, query):
        proj_query = self.linear_query(query)   # affine transformation
        # scaled dot product
        # (N, 1, H) x (N, H, L) -> (N, 1, L)
        dot_products = torch.bmm(proj_query, self.proj_keys.permute(0, 2, 1))
        scores = dot_products / np.sqrt(self.d_k)
        return scores

    def forward(self, query, mask=None):
        # using keys and query to compute "alignment scores":
        scores = self.score_function(query)     # (N, 1, L)
        # make the attention score equal to zero for the padding data points:
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # using alignment scores to compute attention scores:
        alphas = F.softmax(scores, dim=-1)      # (N, 1, L)
        self.alphas = alphas.detach()

        # using values and attention scores to generate the context vector:
        # (N, 1, L) x (N, L, H) -> (N, 1, H)
        context = torch.bmm(alphas, self.values)
        return context


class EncoderDecoderAttn(EncoderDecoder):
    def __init__(self, encoder, decoder, input_len, target_len,
                 teacher_forcing_proba=0.5):
        super().__init__(encoder, decoder, input_len, target_len,
                         teacher_forcing_proba)
        self.alphas = None

    def init_outputs(self, batch_size):
        device = next(self.parameters()).device
        # (N, L(target), F)
        self.outputs = torch.zeros(batch_size, self.target_len,
                                   self.encoder.n_features).to(device)
        # (N, L(target), L(source))
        self.alphas = torch.zeros(batch_size, self.target_len, self.input_len).to(device)

    def store_outputs(self, i, out):
        self.outputs[:, i:i+1, :] = out
        self.alphas[:, i:i+1, :] = self.decoder.attn.alphas


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, input_dim=None, proj_values=True):
        super().__init__()
        self.linear_out = nn.Linear(n_heads * d_model, d_model)
        self.attn_heads = nn.ModuleList([
            Attention(d_model, input_dim=input_dim, proj_values=proj_values)
            for _ in range(n_heads)
            ])

    def init_keys(self, key):
        for attn in self.attn_heads:
            attn.init_keys(key)

    @property
    def alphas(self):
        # shape: (n_heads, N, 1, L (source))
        return torch.stack([attn.alphas for attn in self.attn_heads], dim=0)

    def output_function(self, contexts):
        #  (N, 1, n_heads * D)
        concatenated = torch.cat(contexts, axis=-1)
        # Linear transform to go back to original dimension
        out = self.linear_out(concatenated)     # (N, 1, D)
        return out

    def forward(self, query, mask=None):
        contexts = [attn(query, mask=mask) for attn in self.attn_heads]
        out = self.output_function(contexts)
        return out


class EncoderSelfAttn(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, n_features=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_units = ff_units
        self.n_features = n_features
        self.self_attn_heads = MultiHeadAttention(n_heads, d_model,
                                                  input_dim=n_features)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_units),
                                 nn.ReLU(),
                                 nn.Linear(ff_units, d_model))

    def forward(self, query, mask=None):
        self.self_attn_heads.init_keys(query)   # query <- source data points
        att = self.self_attn_heads(query, mask)  # att <- context vector
        out = self.ffn(att)                     # out <- hidden state
        return out


class DecoderSelfAttn(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, n_features=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_units = ff_units
        self.n_features = d_model if n_features is None else n_features
        self.self_attn_heads = MultiHeadAttention(n_heads, d_model,
                                                  input_dim=self.n_features)
        self.cross_attn_heads = MultiHeadAttention(n_heads, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_units),
                                 nn.ReLU(),
                                 nn.Linear(ff_units, self.n_features))

    def init_keys(self, states):
        self.cross_attn_heads.init_keys(states)

    def forward(self, query, source_mask=None, target_mask=None):
        self.self_attn_heads.init_keys(query)
        att1 = self.self_attn_heads(query, target_mask)
        att2 = self.cross_attn_heads(att1, source_mask)
        out = self.ffn(att2)
        return out


class EncoderDecoderSelfAttn(nn.Module):
    def __init__(self, encoder, decoder, input_len, target_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_len = input_len
        self.target_len = target_len
        self.target_mask = self.subsequent_mask(self.target_len)

    @staticmethod
    def subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = (
                1 - torch.triu(torch.ones(attn_shape), diagonal=1)
                ).bool()
        return subsequent_mask

    def encode(self, source_seq, source_mask):
        # encodes the source sequences and uses the result to init the decoder
        encoder_states = self.encoder(source_seq, source_mask)
        self.decoder.init_keys(encoder_states)

    def decode(self, shifted_target_seq, source_mask=None, target_mask=None):
        # decodes/generates a sequence using the shifted (masked) target sequences
        # JUST in TRAIN mode
        outputs = self.decoder(shifted_target_seq, source_mask, target_mask)
        return outputs

    def predict(self, source_seq, source_mask):
        # decodes/generates a sequence using one input at a time
        # JUST in EVAL mode
        inputs = source_seq[:, -1:]
        for i in range(self.target_len):
            out = self.decode(inputs,
                              source_mask,
                              self.target_mask[:, :i+1, :i+1])
            out = torch.cat([inputs, out[:, -1:, :]], dim=-2)
            inputs = out.detach()
        outputs = inputs[:, 1:, :]
        return outputs

    def forward(self, X, source_mask=None):
        # sends the mask to the same device as the inputs
        self.target_mask = self.target_mask.type_as(X).bool()
        # slices the input to get source sequence
        source_seq = X[:, :self.input_len, :]
        # encodes source sequence AND initializes decoder
        self.encode(source_seq, source_mask)
        if self.training:
            # slices the input to get the shifted target sequence
            shifted_target_seq = X[:, self.input_len-1:-1, :]
            # decodes using the mask to prevent data leaking
            outputs = self.decode(shifted_target_seq,
                                  source_mask,
                                  self.target_mask)
        else:
            # decodes using its own prediction
            outputs = self.predict(source_seq, source_mask)
        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsequeeze(1)
        angular_speed = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(1000.0) / d_model))
        # even dimensions:
        pe[:, 0::2] = torch.sin(position * angular_speed)
        # odd dimensions:
        pe[:, 1::2] = torch.cos(position * angular_speed)

        # use register_buffer() to define an attribute of the module,
        # an attribute is part of the module's state, yet is not parameter
        # so that they will not update by gradient descent
        self.register_buffer("pe", pe.unsequeeze(0))

    def forward(self, X):
        # X is (N, L, D)
        # pe is (1, max_len, D)
        scaled_x = X * np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, :X.size(1), :]
        return encoded


class EncoderPe(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, n_features=None,
            max_len=100):
        super().__init__()
        pe_dim = d_model if n_features is None else n_features
        self.pe = PositionalEncoding(max_len, pe_dim)
        self.layer = EncoderSelfAttn(n_heads, d_model, ff_units, n_features)

    def forward(self, query, mask=None):
        query_pe = self.pe(query)
        out = self.layer(query_pe, mask)    # .layer() equals .layer.forward()
        return out


class DecoderPe(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, n_features=None,
            max_len=100):
        super().__init__()
        pe_dim = d_model if n_features is None else n_features
        self.pe = PositionalEncoding(max_len, pe_dim)
        self.layer = DecoderSelfAttn(n_heads, d_model, ff_units, n_features)

    def init_keys(self, states):
        self.layer.init_keys(states)

    def forward(self, query, source_mask=None, target_mask=None):
        query_pe = self.pe(query)
        out = self.layer(query_pe, source_mask, target_mask)
        return out
