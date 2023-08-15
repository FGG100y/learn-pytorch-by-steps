import numpy as np


n_epochs = 1000  # Defines number of epochs

losses = []
for epoch in range(n_epochs):
    # inner loop (though mini-batches)
    mini_batch_losses = []
    for x_batch, y_batch in train_loader:
        # need to send these mini-batch from CPU to GPU
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # performs one train step and returns the loss
        batch_loss = train_step_fn(x_batch, y_batch)
        mini_batch_losses.append(batch_loss)

    loss = np.mean(mini_batch_losses)
    losses.append(loss)
