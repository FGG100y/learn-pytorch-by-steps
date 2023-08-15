import torch
import numpy as np


def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        # need to send these mini-batch from CPU to GPU
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # performs one train step and returns the loss
        batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(batch_loss)

    return np.mean(mini_batch_losses)


n_epochs = 1000  # Defines number of epochs

losses = []
val_losses = []
for epoch in range(n_epochs):
    # inner loop (though mini-batches)
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    with torch.no_grad():  # NO gradients in validation!
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)

    # records both losses and val_losses for each epoch under tag 'loss'
    writer.add_scalars(
        main_tag="loss",
        tag_scalar_dict={"training": loss, "validation": val_loss},
        global_step=epoch,
    )
writer.close()

save_checkpoint()
