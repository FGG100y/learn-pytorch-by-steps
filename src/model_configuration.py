import torch
import torch.nn as nn
import torch.optim as optim

#  from model.training.utils import make_train_step_fn
#  from model.training.utils import make_val_step_fn


device = "cuda" if torch.cuda.is_available() else "cpu"


lr = 0.1
torch.manual_seed(42)

# create a model and send to device
model = nn.Sequential(nn.Linear(1, 1)).to(device)
print(model.state_dict())

# Defines a SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines a MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

#  # NOTE that these lines below already integrated into MyTrainingClass
#  # create train_step function
#  train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
#
#  # create train_step function
#  val_step_fn = make_val_step_fn(model, loss_fn)
#
#  # create a tensorboard SummaryWriter
#  writer = SummaryWriter('runs/simple_linear_regression')
#  # fetch a single batch so we can use add_graph
#  x_dummy, y_dummpy = next(iter(train_loader))
#  writer.add_graph(model, x_dummy.to(device))
#
#  model, optimizer, *rest = load_checkpoint(model, optimizer)
#  print(model.state_dict())
