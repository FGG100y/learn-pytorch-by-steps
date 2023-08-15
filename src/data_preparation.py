import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from linear_regression_in_numpy import true_model

#  from data.preparation import CustomDataset

#  import matplotlib.pyplot as plt
#
#  plt.style.use("fivethirtyeight")

device = "cuda" if torch.cuda.is_available() else "cpu"
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
