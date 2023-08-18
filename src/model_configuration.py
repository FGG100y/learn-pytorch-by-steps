import torch
import torch.nn as nn
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"


lr = 0.1
torch.manual_seed(42)

classification = True
image_data = True

model = nn.Sequential()

if classification:
    if image_data:
        model.add_module("flatten", nn.Flatten())
        model.add_module("output", nn.Linear(25, 1, bias=False))
        model.add_module("sigmoid", nn.Sigmoid())
        optimizer = optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
    else:
        model.add_module("linear", nn.Linear(2, 1))
        print(model.state_dict())

        optimizer = optim.SGD(model.parameters(), lr=lr)

        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
else:
    #  # create a model and send to device
    #  model = nn.Sequential(nn.Linear(1, 1)).to(device)
    model.add_module("linear", nn.Linear(1, 1))
    print(model.state_dict())

    # Defines a SGD optimizer to update the parameters
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')
