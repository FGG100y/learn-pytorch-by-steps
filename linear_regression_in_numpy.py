import numpy as np
from sklearn.linear_model import LinearRegression


# Data Generation ----------------------------
def gen_data(N=100):
    x, y = true_model(N)

    # Shuffles the indices
    idx = np.arange(N)
    np.random.shuffle(idx)

    # Uses first 80 random indices for train
    train_idx = idx[:int(N*.8)]
    # Uses the remaining indices for validation
    val_idx = idx[int(N*.8):]

    # Generates train and validation sets
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    return x_train, y_train, x_val, y_val


def true_model(N=100):
    true_b = 1
    true_w = 2

    np.random.seed(42)
    x = np.random.rand(N, 1)
    epsilon = (.1 * np.random.randn(N, 1))
    y = true_b + true_w * x + epsilon       # the "true" model
    return x, y


x_train, y_train, x_val, y_val = gen_data()

# -------------------------------------------

# # training model by manual coding (using numpy only)

# Step 0 - Initializes parameters "b" and "w" randomly
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

print(b, w)

# Sets learning rate - this is "eta" ~ the "n"-like Greek letter
lr = 0.1
# Defines number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Step 1 - Computes model's predicted output - forward pass
    yhat = b + w * x_train

    # Step 2 - Computes the loss
    # We are using ALL data points, so this is BATCH gradient
    # descent. How wrong is our model? That's the error!
    error = (yhat - y_train)
    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()

    # Step 3 - Computes gradients for both "b" and "w" parameters
    b_grad = 2 * error.mean()
    w_grad = 2 * (x_train * error).mean()

    # Step 4 - Updates parameters using gradients and the learning rate
    b = b - lr * b_grad
    w = w - lr * w_grad

print(b, w)


# Sanity Check: do we get the same results as our gradient descent?
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])
