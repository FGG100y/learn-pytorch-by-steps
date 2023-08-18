import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


class MyTrainingClass(object):
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
        """to specify a different device
        """
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Couldn't send it to {device}, send it to {self.device}")
            self.model.to(self.device)

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
