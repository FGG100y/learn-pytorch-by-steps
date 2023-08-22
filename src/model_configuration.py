import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"


lr = 0.051
torch.manual_seed(42)

classification = True
image_data = True
using_cnn = True


model = nn.Sequential()

if classification:
    if image_data:
        if using_cnn:
            # featurizer part of a CNN model:
            # Block 1: 1@10x10 -> n_channels@8x8 -> n_channels@4x4
            n_channels = 1
            # convolution:
            # 3x3 filter/kernel 意味着对 10x10 卷积后得到 8x8
            model.add_module(
                "conv1",
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_channels,
                    kernel_size=3,  # 3x3 filter/kernel -> n_channels@8x8
                ),
            ),
            model.add_module("relu1", nn.ReLU()),
            # maxpooling:
            # n_channels@8x8 等于 16 chuncks of 2x2，然后得到 n_channels@4x4
            model.add_module("maxp1", nn.MaxPool2d(kernel_size=2)),
            # Flattening: n_channels * 4 * 4
            model.add_module("flatten", nn.Flatten())

            # classification part
            # hidden layers
            model.add_module(
                "fc1",
                nn.Linear(
                    in_features=n_channels * 4 * 4,
                    out_features=10,
                ),
            )
            model.add_module("relu2", nn.ReLU())
            # output layer
            model.add_module("fc2", nn.Linear(in_features=10, out_features=3))

            # loss and optimizer
            optimizer = optim.SGD(model.parameters(), lr=lr)
            # Since NO nn.LogSoftmax in model's last layer,
            # one must use nn.CrossEntropyLoss(), else using nn.NLLLoss():
            loss_fn = nn.CrossEntropyLoss(reduction="mean")
        else:
            # classification task for image data using simple DNN
            model.add_module("flatten", nn.Flatten())
            model.add_module("hidden0", nn.Linear(25, 5, bias=False))
            model.add_module("activation0", nn.ReLU())
            model.add_module("hidden1", nn.Linear(5, 3, bias=False))
            model.add_module("activation1", nn.ReLU())
            model.add_module("output", nn.Linear(3, 1, bias=False))
            model.add_module("sigmoid", nn.Sigmoid())

            optimizer = optim.SGD(model.parameters(), lr=lr)
            loss_fn = nn.BCELoss()
    else:
        # classification task for other tasks (not image data)
        model.add_module("linear", nn.Linear(2, 1))

        optimizer = optim.SGD(model.parameters(), lr=lr)
        # Since NO nn.Sigmoid() in model's last layer,
        # one must use nn.BCEWithLogitsLoss(), else using nn.BCELoss():
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
else:
    #  # create a model and send to device
    #  model = nn.Sequential(nn.Linear(1, 1)).to(device)
    model.add_module("linear", nn.Linear(1, 1))
    print(model.state_dict())

    # Defines a SGD optimizer to update the parameters
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction="mean")
