import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


# frozen layers
def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False


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


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
