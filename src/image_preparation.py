import torch
from image_classification import generate_dataset
from torch.utils.data import (DataLoader, Dataset, SubsetRandomSampler,
                              TensorDataset, WeightedRandomSampler,
                              random_split)
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip,
                                    Resize, ToPILImage, ToTensor)

from src.data.preparation.utils import TransformedTensorDataset


def index_splitter(n, splits, seed=42):
    idx = torch.arange(n)

    splits_tensor = torch.as_tensor(splits)

    multiplier = n / splits_tensor.sum()
    splits_tensor = (multiplier * splits_tensor).long()

    diff = n - splits_tensor.sum()
    splits_tensor[0] += diff

    torch.manual_seed(seed)
    return random_split(idx, splits_tensor)


def make_balanced_sampler(y):
    _, counts = y.unique(return_counts=True)
    weights = 1.0 / counts.float()
    sample_weights = weights[y.squeeze().long()]
    generator = torch.Generator()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        generator=generator,
        replacement=True,
    )
    return sampler


# --------------------------------
# Generate data and transform them
# --------------------------------

images, labels = generate_dataset(img_size=10, n_images=1000, binary=False, seed=17)
x_tensor = torch.as_tensor(images / 255).float()
y_tensor = torch.as_tensor(labels).long()

train_idx, val_idx = index_splitter(len(x_tensor), [80, 20])

x_train_tensor = x_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]
x_val_tensor = x_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]

augmentation = False
weighted_sampler = True

if augmentation:
    train_composer = Compose(
        [RandomHorizontalFlip(p=0.5), Normalize(mean=(0.5,), std=(0.5,))]
    )
    val_composer = Compose([Normalize(mean=(0.5,), std=(0.5,))])

    train_dataset = TransformedTensorDataset(
        x_train_tensor, y_train_tensor, transform=train_composer
    )
    if weighted_sampler:
        sampler = make_balanced_sampler(y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    val_dataset = TransformedTensorDataset(
        x_val_tensor, y_val_tensor, transform=val_composer
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)
else:
    train_composer = Compose([Normalize(mean=(0.5,), std=(0.5,))])
    val_composer = Compose([Normalize(mean=(0.5,), std=(0.5,))])

    train_dataset = TransformedTensorDataset(
        x_train_tensor, y_train_tensor, transform=None
    )
    val_dataset = TransformedTensorDataset(
        x_val_tensor, y_val_tensor, transform=None
    )

    sampler = make_balanced_sampler(y_train_tensor)
    # NOTE that cannot set shuffle=True when using a sampler
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, sampler=sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8)
