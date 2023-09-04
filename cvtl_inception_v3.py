import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet, inception_v3, resnet18
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import (CenterCrop, Compose, Normalize,
                                    RandomResizedCrop, Resize, ToPILImage,
                                    ToTensor)


from cvtl_utils import freeze_model


model = inception_v3(pretrained=False)
freeze_model(model)

# replacing the layers for both "main" and "auxiliary" classifiers:
torch.manual_seed(42)
model.AuxLogits.fc = nn.Linear(768, 3)
model.fc = nn.Linear(2048, 3)


# standard cross-entropy loss not work here because the Inception model outputs
# two tensors, one for each classifier. But we can create a simple function to
# handle multiple outputs, compute the corresponding losses then return their
# total:
def inception_loss(outputs, labels):
    try:
        main, aux = outputs
    except ValueError:
        main = outputs
        aux = None
        loss_aux = 0

    multi_loss_fn = nn.CrossEntropyLoss(reduction="mean")
    loss_main = multi_loss_fn(main, labels)
    if aux is not None:
        loss_aux = multi_loss_fn(aux, labels)
    return loss_main + 0.4 * loss_aux


optimizer = optim.Adam(model.parameters(), lr=3e-4)
mtc_incep = MyTrainingClass(model, inception_loss, optimizer)

# data loader
normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
composer = Compose([Resize(299),
                    ToTensor(),
                    normalizer])
train_data = ImageFolder(root='data/cv/train', transforms=composer)
val_data = ImageFolder(root='data/cv/test', transforms=composer)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# model training:
mtc_incep.set_loader(train_loader, val_loader)
mtc_incep.train(n_epochs=3)

# evaluation:
MyTrainingClass.loader_apply(val_loader, mtc_incep.correct)
