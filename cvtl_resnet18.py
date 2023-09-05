"""
Transform Learning using resnet: fine-tuning or feature-extraction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.models import alexnet, inception_v3, resnet18
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision.transforms import (CenterCrop, Compose, Normalize,
                                    RandomResizedCrop, Resize, ToPILImage,
                                    ToTensor)

from cvtl_utils import freeze_model, preprocessed_dataset, weights_init


# imagenet statisitcs:
normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

composer = Compose([Resize(256),
                    CenterCrop(224),
                    ToTensor(),
                    normalizer])

# build loader of each set:
train_data = ImageFolder(root='data/cv/train', transforms=composer)
val_data = ImageFolder(root='data/cv/test', transforms=composer)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

num_classes = 3
fine_tuning = True
manually_init_weights = True
model = resnet18(pretrained=False)

if fine_tuning:
    # No freezing layers here, since fine-tuning entails the training of all
    # the weights, not only the "top" layer

    # model configuration:
    torch.manual_seed(42)
    model.fc = nn.Linear(512, num_classes)
    if manually_init_weights:
        with torch.no_grad():
            model.apply(weights_init)
    multi_loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer_model = optim.Adam(model.parameters(), lr=3e-4)

    # model training
    mtc_resnet = MyTrainingClass(model, multi_loss_fn, optimizer_model)
    mtc_resnet.set_loaders(train_loader, val_loader)
    mtc_resnet.train(n_epochs=2)

    # model evaluation
    MyTrainingClass.loader_apply(val_loader, mtc_resnet.correct)
else:   # feature-extraction mode
    # model configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.fc = nn.Identity()
    freeze_model(model)

    # data preparation
    train_preproc = preprocessed_dataset(model, train_loader)
    val_preproc = preprocessed_dataset(model, val_loader)
    train_preproc_loader = DataLoader(train_preproc, batch_size=16, shuffle=True)
    val_preproc_loader = DataLoader(val_preproc, batch_size=16)

    # model configuration -- Top layer model
    torch.manual_seed(42)
    top_model = nn.Sequential(nn.Linear(512, num_classes))
    multi_loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer_model = optim.Adam(top_model.parameters(), lr=3e-4)

    # model training
    mtc_top = MyTrainingClass(top_model, multi_loss_fn, optimizer_model)
    mtc_top.set_loaders(train_preproc_loader, val_preproc_loader)
    mtc_top.train(10)

    # reattach (replacing) the top layer "back"
    # for trying it out on the original dataset (containing the images)
    model.fc = top_model
    mtc_temp = MyTrainingClass(model, None, None)

    # evaluation:
    MyTrainingClass.loader_apply(val_loader, mtc_temp.correct)
