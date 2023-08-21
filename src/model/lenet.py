"""
LeNet-5 using pytorch
"""
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


lenet = nn.Sequential()

# Featurizer

# Block1: 1@28x28 -> 6@28*28 -> 6@14x14
lenet.add_module(
    "C1", nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
)
lenet.add_module("func1", nn.ReLU())
lenet.add_module("S2", nn.MaxPool2d(kernel_size=2))

# Block2: 6@14x14 -> 16@10x10 -> 16@5x5
lenet.add_module(
    "C3", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
)
lenet.add_module("func2", nn.ReLU())
lenet.add_module("S4", nn.MaxPool2d(kernel_size=2))

# Block3: 16@5x5 -> 120@1x1
lenet.add_module(
    "C5", nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
)
lenet.add_module("func2", nn.ReLU())

# Flattening
lenet.add_module("flatten", nn.Flatten())


# Classification

# Hidden Layer
lenet.add_module("F6", nn.Linear(in_features=120, out_features=84))
lenet.add_module("func3", nn.ReLU())
# Output Layer
lenet.add_module("OUTPUT", nn.Linear(in_features=84, out_features=10))
