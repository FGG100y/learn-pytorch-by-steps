import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pathlib
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import alexnet, inception_v3, resnet18
#  from torchvision.models.alexnet import model_urls
from torch.hub import load_state_dict_from_url
from torchvision.transforms import (CenterCrop, Compose, Normalize,
                                    RandomResizedCrop, Resize, ToPILImage,
                                    ToTensor)


net = alexnet(pretrained=False)
print(net)
print()
print(net.classifier)


def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


freeze_model(net)

# replacing the "top" of the model
classes = 3
net.classifier[6] = nn.Linear(4096, classes)

# more configuration
torch.manual_seed(17)
multi_loss_fn = nn.CrossEntropyLoss(reduction="mean")
optimizer_alex = optim.Adam(net.parameters(), lr=3e-4)


# prepare dataset for transfer-learning
# Since using a pre-trained model, we need to use the statistics of the
# original dataset used to train that model. (most likely the ILSVRC dataset)

normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

composer = Compose([Resize(256),
                    CenterCrop(224),
                    ToTensor(),
                    normalizer])

img_dir = pathlib.Path('dataset/vision')
train_data = ImageFolder(root=img_dir / 'train', transform=composer)
val_data = ImageFolder(root=img_dir / 'val', transform=composer)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# model training
mtc_alex = MyTrainingClass(net, multi_loss_fn, optimizer_alex)
mtc_alex.set_loader(train_loader, val_loader)
mtc_alex.train(n_epochs=3)

# evaluating model
MyTrainingClass.loader_apply(val_loader, mtc_alex.correct)
