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


def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


net = alexnet(pretrained=False)
freeze_model(net)

# replacing the "top" of the model
num_classes = 3
net.classifier[6] = nn.Linear(4096, num_classes)

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

# transfer learning the better (time/money saving) way
# NOTE that since all layers but the last are frozen, the output of the second
# -to-last layer is always the same (Assuming No data augmentation).
# NOTE also that the frozen layers simply generating features that will be the
# input of the trainable layers, why not treat the frozen layers as such? We
# can do it in four easy steps:
#   1. keep only the frozen layers in the model
#   2. run the whole dataset through it and collect its outputs as a dataset of
#   features
#   3. train a separate model (that corresponding to he "top" of the original
#   model)
#   4. attach the trained model to the top of the frozen layers
#
# This way, we're effectively splitting the feature extraction and actual
# training phases, thus avoiding the overhead of generating features over and
# over again for every single forward pass.


def preprocessed_dataset(model, loader, device=None):
    """loops over the mini-batches from data loader,
    sends them through the feature extractor model,
    combines the outputs with the corresponding labels,
    and returns a TensorDataset
    """
    if device is None:
        device = next(model.parameters()).device
    features = None
    labels = None

    for i, (x, y) in enumerate(loader):
        model.eval()
        output = model(x.to(device))
        if i == 0:
            features = output.detach().cpu()
            labels = y.cpu()
        else:
            features = torch.cat([features, output.detach().cpu()])
            labels = torch.cat([labels, y.cpu()])

    dataset = TensorDataset(features, labels)
    return dataset


net = alexnet(pretrained=False)
freeze_model(net)

# "removing" the "top" layer by replacing it with identity():
net.classifier[6] = nn.Identity()
print(net.classifier)

# data preparation (1)
train_preproc = preprocessed_dataset(net, train_loader)
val_preproc = preprocessed_dataset(net, val_loader)

# save to file
torch.save(train_preproc.tensors, "img_preproc.pth")
torch.save(val_preproc.tensors, "img_val_preproc.pth")

#  # build datasets later:
#  x, y = torch.load("img_preproc.pth")
#  train_preproc = TensorDataset(x, y)
#  val_preproc = TensorDataset(*torch.load("img_val_preproc.pth"))

# data preparation (2)
train_preproc_loader = DataLoader(train_preproc, batch_size=16, shuffle=True)
val_preproc_loader = DataLoader(val_preproc, batch_size=16)

# top model
top_model = nn.Sequential(nn.Linear(4096, num_classes))
multi_loss_fn = nn.CrossEntropyLoss(reduction="mean")
optimizer_alex = optim.Adam(top_model.parameters(), lr=3e-4)

# model training
mtc_top = MyTrainingClass(net, multi_loss_fn, optimizer_alex)
mtc_top.set_loader(train_preproc_loader, val_preproc_loader)
mtc_top.train(n_epochs=10)

# replacing "back" the top layer
net.model.classifier[6] = top_model
print(net.model.classifier)
