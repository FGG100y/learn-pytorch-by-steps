import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torch.nn.utils import rnn as rnn_utils
from src.data_generation.square_sequences import generate_sequences


# Data Preparation
points, directions = generate_sequences(n=128, seed=42)
test_points, test_directions = generate_sequences(seed=43)

train_data = TensorDataset(torch.as_tensor(points).float(),
                           torch.as_tensor(directions).view(-1, 1).float())

test_data = TensorDataset(torch.as_tensor(test_points).float(),
                          torch.as_tensor(test_directions).view(-1, 1).float())

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)


# model configuration
torch.manual_seed(21)
model = SquareModelLSTM(n_features=2, hidden_dim=2, n_outputs=1)
loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# model training
mtc_rnn = MyTrainingClass(model, loss, optimizer)
mtc_rnn.set_loaders(train_loader, test_loader)
mtc_rnn.train(n_epochs=10)

# plots
viz = False
if viz:
    fig = mtc_rnn.plot_losses()

# evaluation
print(MyTrainingClass.loader_apply(test_loader, mtc_rnn.correct))
