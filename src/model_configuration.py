import torch
import torch.nn as nn
import torch.optim as optim

from model.training.utils import make_train_step_fn
from model.training.utils import make_val_step_fn


device = "cuda" if torch.cuda.is_available() else "cpu"


lr = 0.1
torch.manual_seed(42)

# create a model and send to device
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# Defines a SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines a MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

# create train_step function
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

# create train_step function
val_step_fn = make_val_step_fn(model, loss_fn)
