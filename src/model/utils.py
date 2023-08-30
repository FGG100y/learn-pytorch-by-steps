import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.transforms import Normalize
from torch.utils.tensorboard import SummaryWriter


class MyTrainingClass(object):
    # class attributes
    # for forward hooks
    visualization = {}
    handles = {}
    _gradients = {}
    _parameters = {}
    # for learning rate
    scheduler = None
    is_batch_lr_scheduler = False
    learning_rates = []

    # the constructor
    def __init__(self, model, loss_fn, optimizer):
        # storing arguments as attributes
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # send the model to device
        self.model.to(self.device)

        # placeholder attributes
        self.train_loader = None
        self.val_loader = None
        self.writer = None

        # those that are computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # functions for model, loss function and optimizer
        # NOTE that these functions use attributes as ARGS
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        """to specify a different device"""
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Couldn't send it to {device}, send it to {self.device}")
            self.model.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder="runs"):
        suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.writer = SummaryWriter(f"{folder}/{name}_{suffix}")

    # protected methods (prefix with underscore):
    # methods should not be called by the user, they are supposed to be called
    # either internally or by the child class.
    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):
            # sets model to TRAIN mode:
            self.model.train()

            # Step 1 computes model's predicted output - forward pass
            yhat = self.model(x)

            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)

            # Step 3 - Computes gradients for both "b" and "w" parameters
            loss.backward()

            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            # sets model to EVAL mode:
            self.model.eval()

            # Step 1 computes model's predicted output - forward pass
            yhat = self.model(x)

            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)

            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            # need to send these mini-batch from CPU to GPU
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # performs one train step and returns the loss
            batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(batch_loss)
        loss = np.mean(mini_batch_losses)

        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.daterministric = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def train(self, n_epochs, seed=42):
        self.set_seed(seed)

        for epoch in range(n_epochs):
            self.total_epochs += 1

            # inner loop (though mini-batches)
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            with torch.no_grad():  # NO gradients in validation!
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            if self.writer:
                scalars = {"training": loss}
                if val_loss is not None:
                    scalars.update({"validation": val_loss})
                # records both losses for each epoch under tag 'loss'
                self.writer.add_scalars(
                    main_tag="loss",
                    tag_scalar_dict=scalars,
                    global_step=epoch,
                )
        if self.writer:
            # flushs the writer
            self.writer.flush()

    def save_checkpoint(self, filename="models/model_checkpoint.pth"):
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.losses,
            "val_loss": self.val_losses,
        }
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_epochs = checkpoint["epoch"]
        self.losses = checkpoint["loss"]
        self.val_losses = checkpoint["val_loss"]
        self.model.train()  # for resuming training

    def predict(self, x):
        self.model.eval()  # inference mode
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()  # back to train mode

        # Detach it, bring it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label="Training Loss", c="b")
        if self.val_loader:
            plt.plot(self.val_losses, label="Validation Loss", c="r")
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        if self.train_loader and self.writer:
            x_dummy, y_dummpy = next(iter(self.train_loader))
            self.writer.__add_graph(self.model, x_dummy.to(self.device))

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # staticmethod, what?
    # The @staticmethod decorator allows the method to be called on an
    # uninstanciated class object. It is as if we're attaching a method to a
    # class but that method does not dependend on an instance of that class.
    # A staticmethod does not have a "self" argument. The inner workings of the
    # function must be indenpendent of the instance of the class it belongs to.
    # The staticmethod can be executed from the class itself instead of from
    # one of its instances.
    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None, layer_name="", title=None):
        # the number of images is the number of subplots in a row
        n_images = len(axs)
        # gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        # for each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # set title, labels, and removes ticks
            if title is not None:
                ax.set_title(f"{title} #{j}", fontsize=12)
            shp = np.atleast_2d(image).shape
            ax.set_ylabel(f"{layer_name}\n{shp[0]}x{shp[1]}", rotation=0, labelpad=40)
            xlabel1 = "" if y is None else f"\nLabel: {y[j]}"
            xlabel2 = "" if yhat is None else f"\nPredicted: {yhat[j]}"
            xlabel = f"{xlabel1}{xlabel2}"
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])

            # Plots weight as an image
            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap="gray",
                vmin=minv,
                vmax=maxv,
            )
        return

    def visualize_filters(self, layer_name, **kwargs):
        try:
            # get the layer object from the model
            layer = self.model
            for name in layer_name.split("."):
                layer = getattr(layer, name)
            # only looking at filters for 2D convolutions
            if isinstance(layer, nn.Conv2d):
                # takes the weight information
                weights = layer.weight.data.cpu().numpy()
                # weight -> (channels_out (filter), channels_in, H, W)
                n_filters, n_channels, _, _ = weights.shape

                # builds a figure
                size = (2 * n_channels + 2, 2 * n_filters)
                fig, axes = plt.subplots(n_filters, n_channels, figsize=size)
                axes = np.atleast_2d(axes)
                axes = axes.reshape(n_filters, n_channels)
                # for each channels_out (filter)
                for i in range(n_filters):
                    MyTrainingClass._visualize_tensors(
                        axes[i, :],
                        weights[i],
                        layer_name=f"Filter ${i}",
                        title="Channel",
                    )
                for ax in axes.flat:
                    ax.label_outer()
                fig.tight_layout()
                return fig
        except AttributeError:
            return

    def attach_hooks(self, layers_to_hook, hook_fn=None):
        # clear any prevois values
        self.visualation = {}
        # create the dictionary to map layer objects to thier names
        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules[1:]}

        if hook_fn is None:

            def hook_fn(layer, inputs, outputs):
                name = layer_names[layer]
                values = outputs.detach().cpu().numpy()
                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate(
                        self.visualization[name], values
                    )

        for name, layer in modules:
            # if the layer is in our list
            if name in layers_to_hook:
                # initializes the corresponding key in the dictionary
                self.visualization[name] = None
                self.handles[name] = layer.register_forward_hook(hook_fn)

    def remove_hooks(self):
        # loop throngh all hooks and removes them
        for handle in self.handles.values():
            handle.remove()
        # clear the dict
        self.handles = {}

    def visualize_outputs(self, layers, n_images=10, y=None, yhat=None):
        layers = filter(lambda lay: lay in self.visualization.keys(), layers)
        layers = list(layers)
        shapes = [self.visualization[layer].shape for layer in layers]
        n_rows = [shape[1] if len(shape) == 4 else 1 for shape in shapes]
        total_rows = np.sum(n_rows)
        fig, axes = plt.subplots(
            total_rows, n_images, figsize=(1.5 * n_images, 1.5 * total_rows)
        )
        axes = np.atleast_2d(axes).reshape(total_rows, n_images)
        # Loops through the layers, one layer per row of subplots
        row = 0
        for i, layer in enumerate(layers):
            start_row = row
            # Takes the produced feature maps for that layer
            output = self.visualization[layer]
            is_vector = len(output.shape) == 2
            for j in range(n_rows[i]):
                MyTrainingClass._visualize_tensors(
                    axes[row, :],
                    output if is_vector else output[:, j].squeeze(),
                    y,
                    yhat,
                    layer_name=(
                        layers[i] if is_vector else f"{layers[i]}\nfil#{row-start_row}"
                    ),
                    title="Image" if (row == 0) else None,
                )
                row += 1
        for ax in axes.flat:
            ax.label_outer()
        plt.tight_layout()
        return fig

    def correct(self, x, y, threshold=0.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()

        # size of batch, number of classes
        n_samples, n_dims = yhat.shape
        if n_dims > 1:
            # in a multiclass classification, the largest logit always wins,
            # so we don't bother getting probabilities

            # PyTorch's version of argmax -> (max_value, index of max_value)
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            # in binary classification, we need to check if the last layer is
            # a sigmoid (and then it produces proba.)
            if isinstance(self.model, nn.Sequential) and isinstance(
                self.model[-1], nn.Sigmoid
            ):
                predicted = (yhat > threshold).long()
            # or something else (logits), which need to convert using sigmoid
            else:
                predicted = (torch.sigmoid(yhat) > threshold).long()

        # how many samples got classified correctly for each class
        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result)

    @staticmethod
    def loader_apply(loader, func, reduce="sum"):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        results = torch.stack(results, axis=0)
        if reduce == "sum":
            results = results.sum(axis=0)
        elif reduce == "mean":
            results = results.float().mean(axis=0)
        return results

    @staticmethod
    def statistics_per_channel(images, labels):
        # NCHW format
        n_samples, n_channels, n_height, n_weight = images.size()
        # flatten HW into a single dimension
        flatten_per_channel = images.reshape(n_samples, n_channels, -1)

        # computes stats of each image per channel
        # average/std pixel value per channel
        # (n_samples, n_channels)
        means = flatten_per_channel.mean(axis=2)
        stds = flatten_per_channel.std(axis=2)

        # add up stats of all images in a mini-batch
        sum_means = means.sum(axis=0)
        sum_stds = stds.sum(axis=0)

        # make tensor of shape (1, n_channels)
        n_samples = torch.tensor([n_samples] * n_channels).float()

        return torch.stack([n_samples, sum_means, sum_stds], axis=0)

    @staticmethod
    def make_normalizer(loader):
        total_samples, total_means, total_stds = MyTrainingClass.loader_apply(
            loader, MyTrainingClass.statistics_per_channel
        )
        norm_mean = total_means / total_samples
        norm_std = total_stds / total_samples

        return Normalize(mean=norm_mean, std=norm_std)

    def lr_range_test(
        self, data_loader, end_lr, num_iter=100, step_mode="exp", alpha=0.05, ax=None
    ):
        # the test updates both model and optimizer, so we need to store their
        # initial states to restore them in the end
        previous_states = {
            "model": deepcopy(self.model.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
        }
        # retrieves the learning rate set in the optimizer
        start_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        # builds a custom function and corresponding scheduler
        lr_fn = make_lr_fn(start_lr, end_lr, num_iter, step_mode="exp")
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)
        # variables for tracking results and iterations
        tracking = {"loss": [], "lr": []}
        iteration = 0

        # if there are iterations than mini-batches in the data loader,
        # it will have to loop over it more than once
        while iteration < num_iter:
            # typical mini-batch inner loop:
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # step 1
                yhat = self.model(x_batch)
                # step 2
                loss = self.loss_fn(yhat, y_batch)
                # step 3
                loss.backward()

                # here we keep track of the losses and lr
                tracking["lr"].append(scheduler.get_last_lr()[0])
                if iteration == 0:
                    tracking["loss"].append(loss.item())
                else:
                    prev_loss = tracking["loss"][-1]
                    smoothed_loss = alpha * loss.item() + (1 - alpha) * prev_loss
                    tracking["loss"].append(smoothed_loss)
                iteration += 1
                # number of iterations reached:
                if iteration == num_iter:
                    break

                # step 4
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

        # restore the original states
        self.optimizer.load_state_dict(previous_states["optimizer"])
        self.model.load_state_dict(previous_states["model"])

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig = ax.get_figure()
        ax.plot(tracking["lr"], tracking["loss"])
        if step_mode == "exp":
            ax.set_xscale("log")
            ax.set_xlabel("learning rate")
            ax.set_ylabel("loss")
            fig.tight_layout()
        return tracking, fig

    def capture_gradients(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]

        #  modules = list(self.model.named_modules())
        self._gradients = {}

        def make_log_fn(name, param_id):
            def log_fn(grad):
                self._gradients[name][param_id].append(grad.tolist())
                return None

            return log_fn

        for name, layer in self.model.named_modules():
            if name in layers_to_hook:
                self._gradients.update({name: {}})
                for param_id, p in layer.named_modules():
                    if p.requires_grad:
                        self._gradients[name].update({param_id: []})
                        log_fn = make_log_fn(name, param_id)
                        self.handles[f"{name}.{param_id}.grad"] = p.register_hook(
                            log_fn
                        )
        return

    def capture_parameters(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]

        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules}
        self._parameters = {}

        for name, layer in modules:
            if name in layers_to_hook:
                self._parameters.update({name: {}})
                for param_id, p in layer.named_modules():
                    self._parameters[name].update({param_id: []})

        def fw_hook_fn(layer, inputs, outputs):
            name = layer_names[layer]
            for param_id, parameter in layer.named_parameters():
                self._parameters[name][param_id].append(parameter.tolist())

        self.attach_hooks(layers_to_hook, fw_hook_fn)
        return

    def set_lr_scheduler(self, scheduler):
        # makes sure the scheduler in the argument is assigned to the optimizer
        if scheduler.optimizer == self.optimizer:
            self.scheduler = scheduler
            if (
                isinstance(scheduler, optim.lr_scheduler.CyclicLR)
                or isinstance(scheduler, optim.lr_scheduler.OneCycleLR)
                or isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts)
            ):
                self.is_batch_lr_scheduler = True
            else:
                self.is_batch_lr_scheduler = False

    def _epoch_schedulers(self, val_loss):
        if self.scheduler:
            if not self.is_batch_lr_scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = list(
                    map(
                        lambda d: d["lr"],
                        self.scheduler.optimizer.state_dict()["param_groups"],
                    )
                )
                self.learning_rates.append(current_lr)
