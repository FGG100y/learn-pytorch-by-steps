import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from linear_regression_in_numpy import gen_data

#  from src.data.preparation import CustomDataset

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

device = "cuda" if torch.cuda.is_available() else "cpu"
x_train, y_train, x_val, y_val = gen_data()

# Our data was in Numpy arrays, but we need to transform them
# into PyTorch's Tensors
x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()

train_data = CustomDataset(x_train_tensor, y_train_tensor)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True,
)
#  next(iter(train_loader))
