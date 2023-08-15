import torch


def make_train_step_fn(model, loss_fn, optimizer):
    # build function that performs a step in the train loop
    def perform_train_step_fn(x, y):
        # sets model to TRAIN mode:
        model.train()

        # Step 1 computes model's predicted output - forward pass
        yhat = model(x)

        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)

        # Step 3 - Computes gradients for both "b" and "w" parameters
        loss.backward()

        # Step 4 - Updates parameters using gradients and the learning rate
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    return perform_train_step_fn


def make_val_step_fn(model, loss_fn):
    # build function that performs a step in the validation loop
    def perform_val_step_fn(x, y):
        # sets model to EVAL mode:
        model.eval()

        # Step 1 computes model's predicted output - forward pass
        yhat = model(x)

        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)

        return loss.item()

    return perform_val_step_fn


def save_checkpoint():
    checkpoint = {
        "epoch": n_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, "models/model_checkpoint.pth")


def load_checkpoint(model, optimizer):
    checkpoint = torch.load("models/model_checkpoint.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    saved_epoch = checkpoint["epoch"]
    saved_losses = checkpoint["loss"]
    saved_val_losses = checkpoint["val_loss"]
    return model, optimizer, saved_epoch, saved_losses, saved_val_losses
