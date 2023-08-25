import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def make_lr_fn(start_lr, end_lr, num_iter, step_mode="exp"):
    """
    The general idea is pretty much the same as the grid search:
    It tries multiple learning rates and logs the corresponding losses.
    But the difference is: It evaluates the loss over a single mini-batch, and
    then changes the learning rate before moving on to the next mini-batch.
    """
    if step_mode == "linear":
        factor = (end_lr / start_lr - 1) / num_iter

        def lr_fn(iteration):
            return 1 + iteration * factor

    else:
        factor = (np.log(end_lr) - np.log(start_lr)) / num_iter

        def lr_fn(iteration):
            return np.exp(factor) ** iteration

    return lr_fn


model_cnn = CNN(n_feature=5, p=0.3)
loss_fn = nn.CrossEntropyLoss(reduction="mean")
#  optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=3e-4)
start_lr = 0.01
end_lr = 0.1
num_iter = 10
lr_fn = make_lr_fn(start_lr, end_lr, num_iter, step_mode="exp")
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=start_lr)
lr_scheduler = LambdaLR(optimizer_cnn, lr_lambda=lr_fn)
