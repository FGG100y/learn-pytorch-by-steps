import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from linear_regression_in_numpy import gen_data
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

device = "cuda" if torch.cuda.is_available() else "cpu"
x_train, y_train, x_val, y_val = gen_data()

# Our data was in Numpy arrays, but we need to transform them
# into PyTorch's Tensors and then we send them to the chosen device
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
