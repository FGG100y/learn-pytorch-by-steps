import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from linear_regression_in_numpy import true_model
from logicstic_regression_dataset import moon_data

classification = True

if classification:
    X_train, X_val, y_train, y_val = moon_data()
    # Builds tensors from Numpy arrays
    x_train_tensor = torch.as_tensor(X_train).float()
    y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()
    x_val_tensor = torch.as_tensor(X_val).float()
    y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()
    # Builds dataset containing ALL data points
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    val_data = TensorDataset(x_val_tensor, y_val_tensor)
else:
    x, y = true_model()

    # Our data was in Numpy arrays, but we need to transform them
    # into PyTorch's Tensors
    x_tensor = torch.as_tensor(x).float()
    y_tensor = torch.as_tensor(y).float()

    dataset = TensorDataset(x_tensor, y_tensor)

    ratio = 0.8
    n_total = len(dataset)
    n_train = int(n_total * ratio)
    n_val = n_total - n_train
    train_data, val_data = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True,
)

val_loader = DataLoader(dataset=val_data, batch_size=16)
