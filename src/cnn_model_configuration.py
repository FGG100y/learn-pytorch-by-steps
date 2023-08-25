import torch
import torch.nn as nn
import torch.optim as optim


model_cnn = CNN(n_feature=5, p=0.3)
loss_fn = nn.CrossEntropyLoss(reduction="mean")
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=3e-4)
