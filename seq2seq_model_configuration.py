import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from src.data_generation.square_sequences import generate_sequences


def encoder_decoder_demo(teacher_forcing_proba=0.5, verbose=False):
    # a square
    full_seq = torch.tensor([
        [-1, -1], [-1, 1], [1, 1], [1, -1]
        ]).float().view(1, 4, 2)  # (N, L, F)
    source_seq = full_seq[:, :2]  # first two conners
    target_seq = full_seq[:, 2:]  # last two conners

    torch.manual_seed(21)
    encoder = Encoder(n_features=2, hidden_dim=2)  # untrained model
    hidden_seq = encoder(source_seq)  # output shape (N, L, F)
    hidden_final = hidden_seq[:, -1:]
    if verbose == 2:
        print("-*-encoder-*-")
        print(f"the final hidden state:\n{hidden_final}\n")

    decoder = Decoder(n_features=2, hidden_dim=2)
    # initial hidden state is encoder's final hidden state
    decoder.init_hidden(hidden_seq)
    # initial data point is the last element of source sequence
    inputs = source_seq[:, -1:]

    if verbose == 2:
        #  teacher_forcing_proba = 0.5
        target_len = target_seq.shape[1]
        for i in range(target_len):
            print("-*-decoder-*-")
            print(f"Hidden: {decoder.hidden}")
            out = decoder(inputs)
            print(f"Output: {out}\n")
            # the predictions are next step's inputs
            # NOTE the "teacher forcing":
            # model trained using teacher forcing will minimize the loss given the
            # correct inputs at every step of the target sequence. But this will
            # never be the case ate testing time! And we must seek help from stats.
            if torch.rand(1) <= teacher_forcing_proba:
                inputs = target_seq[:, i:i+1]
            else:
                inputs = out

    encdec = EncoderDecoder(encoder, decoder, input_len=2, target_len=2,
                            teacher_forcing_proba=0.5)
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
train_loader = DataLoader(train_data, batch_size=16, shuffle=True,
                          generator=generator)  # ensure reproducibility
test_loader = DataLoader(test_data, batch_size=16)

# model configuration
torch.manual_seed(23)
encoder = Encoder(n_features=2, hidden_dim=2)
decoder = Decoder(n_features=2, hidden_dim=2)
model = EncoderDecoder(encoder, decoder, input_len=2, target_len=2,
                       teacher_forcing_proba=0.5)
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# model training
mtc_seq = MyTrainingClass(model, loss, optimizer)
mtc_seq.set_loaders(train_loader, test_loader)
mtc_seq.train(100)

viz = 1
if viz:
    fig = mtc_seq.plot_losses()
    fig = sequence_pred(mtc_seq, full_test, test_directions)
