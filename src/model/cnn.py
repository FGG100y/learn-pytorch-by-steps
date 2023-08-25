from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_filters, p=0.0):
        super(CNN, self).__init__()
        self.n_filters = n_filters
        self.p = p
        # create the convolution layers
        self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=n_filters,
                kernel_size=3
                )
        self.conv2 = nn.Conv2d(
                in_channels=n_filters,
                out_channels=n_filters,
                kernel_size=3
                )
        # create the linear layers
        # 50 is hardcode magic number
        # 3 is the number of classes
        self.fc1 = nn.Linear(n_filters * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 3)
        # create dropout layers
        self.dropout = nn.Dropout(self.p)

        def featurizer(self, x):
            # first convolutional block
            # 3@28x28 -> n_filters@26x26 -> n_filters@13x13
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2)
            # second convolution block
            # n_filters@13x13 -> n_filters@11x11 -> n_filters@5x5
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2)
            # input dims (n_filters@5x5) output dims (n_filters * 5 * 5)
            x = nn.Flatten()(x)
            return x

        def classifier(self, x):
            # hidden layer
            # input dims (n_feature * 5 * 5), output dims (50)
            if self.p > 0:
                x = self.dropout(x)
            x = self.fc1(x)
            x = F.relu(x)

            # output layer
            # input dims (50), output dim (3)
            if self.p > 0:
                x = self.dropout(x)
            x = self.fc2(x)

            return x

        def forward(self, x):
            x = self.featurizer(x)
            x = self.classifier(x)
            return x
