import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from src.data_generation.square_sequences import generate_sequences


def calc_alphas(ks, q):
    """calculate attention scores"""
    dims = q.size(-1)
    # (N, 1, H) x (N, H, L) -> (N, 1, L)
    products = torch.bmm(q, ks.permute(0, 2, 1))
    scaled_products = products / np.sqrt(dims)
    alphas = F.softmax(scaled_products, dim=-1)
    return alphas


def attention_q_k_v_demo(teacher_forcing_proba=0.5, verbose=False):
    # a square (points with coordinates and directions as sequences)
    full_seq = (
        torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]).float().view(1, 4, 2)
    )  # (N, L, F)
    source_seq = full_seq[:, :2]  # first two conners
    target_seq = full_seq[:, 2:]  # last two conners

    # context vector = \sum alignment_vector

    # alignment_vector: the resulting mulitiplication of a "Value" by its
    # corresponding attention_score

    # attention_score(alphas) are based on matching each hidden state of the
    # decoder to every hidden state of the encoder (good match, high
    # attention-score)

    # context vector will be concatenated to "Query"(hidden state of decoder)
    # and use it as the input for the linear layer that generate predictions

    torch.manual_seed(21)
    encoder = Encoder(n_features=2, hidden_dim=2)  # untrained model
    hidden_seq = encoder(source_seq)  # output shape (N, L, F)
    # hidden states are both "Values" and "Keys":
    keys = hidden_seq
    values = hidden_seq

    decoder = Decoder(n_features=2, hidden_dim=2)
    # initial hidden state is encoder's final hidden state
    decoder.init_hidden(hidden_seq)
    # initial data point is the last element of source sequence
    inputs = source_seq[:, -1:]

    # the first "Query" is the decoder's hidden state (always sequence-first)
    query = decoder.hidden.permute(1, 0, 2)  # N, 1, H

    alphas = calc_alphas(keys, query)
    context_vec = torch.bmm(alphas, values)

    concatenated = torch.cat([context_vec, query], axix=-1)
    if verbose:
        print(concatenated)

    encdec = EncoderDecoder(
        encoder, decoder, input_len=2, target_len=2, teacher_forcing_proba=0.5
    )
    # model expects the full sequence
    # so it can randomly use the teacher forcing:
    encdec.train()
    if verbose:
        print(encdec(full_seq))

    # in evaluation/test mode, it only needs the source sequence as input:
    encdec.eval()
    if verbose:
        print(encdec(source_seq))


demo = 0
if demo:
    encoder_decoder_demo(verbose=1)


# data preparation
points, directions = generate_sequences(n=256, seed=13)
full_train = torch.as_tensor(points).float()  # full sequences as features (X)
target_train = full_train[:, 2:]  # labels (y) for MSELoss

test_points, test_directions = generate_sequences(seed=19)
full_test = torch.as_tensor(test_points).float()
source_test = full_test[:, :2]  # source sequences as features (X)
target_test = full_test[:, 2:]  # target sequences as labels (y)

train_data = TensorDataset(full_train, target_train)
test_data = TensorDataset(source_test, target_test)

generator = torch.Generator()
train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True, generator=generator
)  # ensure reproducibility
test_loader = DataLoader(test_data, batch_size=16)

# model configuration
torch.manual_seed(23)
encoder_self_attn = EncoderSelfAttn(n_heads=3, d_model=2, ff_units=10, n_features=2)
decoder_self_attn = DecoderSelfAttn(n_heads=3, d_model=2, ff_units=10, n_features=2)
model = EncoderDecoderSelfAttn(
    encoder_self_attn, decoder_self_attn, input_len=2, target_len=2
)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# model training
mtc_seq = MyTrainingClass(model, loss_fn, optimizer)
mtc_seq.set_loaders(train_loader, test_loader)
mtc_seq.train(100)

viz = 1
if viz:
    fig = mtc_seq.plot_losses()
    fig = sequence_pred(mtc_seq, full_test, test_directions)
