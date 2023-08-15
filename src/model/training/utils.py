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
