import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from src.data_generation.square_sequences import generate_sequences

# Data Preparation
variable_length_dataset = True
conv1d = False
if variable_length_dataset:
    var_points, var_directions = generate_sequences(variable_len=True)
    var_test_points, var_test_directions = generate_sequences(
        n=64, variable_len=True, seed=15
    )
    train_var_data = CustomDataset(var_points, var_directions)
    test_var_data = CustomDataset(var_test_points, var_test_directions)
    # pack the mini-batch using collate function:
    train_loader = DataLoader(
        train_var_data, batch_size=16, shuffle=True, collate_fn=pack_collate
    )
    test_loader = DataLoader(test_var_data, batch_size=16, collate_fn=pack_collate)
else:
    points, directions = generate_sequences(n=128, seed=13)
    test_points, test_directions = generate_sequences(seed=19)

    if conv1d:
        # sequence-last (NFL) shape:
        train_data = TensorDataset(
            torch.as_tensor(points).float().permute(0, 2, 1),
            torch.as_tensor(directions).view(-1, 1).float()
        )

        test_data = TensorDataset(
            torch.as_tensor(test_points).float().permute(0, 2, 1),
            torch.as_tensor(test_directions).view(-1, 1).float(),
        )
    else:
        train_data = TensorDataset(
            torch.as_tensor(points).float(), torch.as_tensor(directions).view(-1, 1).float()
        )

        test_data = TensorDataset(
            torch.as_tensor(test_points).float(),
            torch.as_tensor(test_directions).view(-1, 1).float(),
        )

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)


# model configuration
if conv1d:
    torch.manual_seed(21)
    model = nn.Sequential()
    model.add_module("conv1d", nn.Conv1d(in_channels=2,
                                         out_channels=1,
                                         kernel_size=2))
    model.add_module("relu", nn.ReLU())
    model.add_module("flatten", nn.Flatten())
    model.add_module("output", nn.Linear(3, 1))
    #  print(model.state_dict())
else:
    torch.manual_seed(21)
    model = SquareModelOne(n_features=2, hidden_dim=2, n_outputs=1,
                           rnn_layer=nn.LSTM, num_layers=1, bidirectional=True
                           )
    #  print(model.state_dict())

loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# model training
mtc_rnn = MyTrainingClass(model, loss, optimizer)
mtc_rnn.set_loaders(train_loader, test_loader)
mtc_rnn.train(n_epochs=100)

# plots
viz = False
if viz:
    fig = mtc_rnn.plot_losses()

# evaluation
print(MyTrainingClass.loader_apply(test_loader, mtc_rnn.correct))
