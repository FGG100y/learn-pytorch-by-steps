# DNN 101

Shapes

The shapes topic of images, sequences:

1. NCHW shape for images:
    - N stands for the Number of images (e.g., in a mini-batch)
    - C stands for the number of Channels (or filters) in each image
    - H stands for each image's Height
    - W stands for each image's Width

2. NCL/NFL shape for sequences:
    - N stands for the Number of sequences (e.g., in a mini-batch)
    - C stands for the number of Channels (or filters) in each sequence
    - L stands for the Length of each sequence
    - F stands for the number of features of each sequence

NOTE that since 1D convolutions only move along the sequence, each feature is
considered an input channel. So, the shape NCL is the same as NFL.

|               | Shape         | Use Case                                  |
| ------------- | ------------- | -------                                   |
| Batch-first   | N,L,F         | Typical shape; RNNs with batch_first=True |
| RNN-friendly  | L,N,F         | Default for RNNs (batch_first=False)      |
| Sequence-last | N,F,L         | Default for 1D convolution                |


# how this scripts work

1. linear_regression_in_numpy.py, linear_regression_in_pytorch.py

```sh
python3 linear_regression_in_*.py
```

2. scripts in src/

These scripts should be viewed as jupyter cell code (which need an shared
namespace of all the functions/modules. I mean that:

a. Get into jupyterlab, jupyter notebook, or IPython (if you like using shell)
b. In IPython console, run these commands in order (-i makes shared namespcace):

For simple linear regression:
```python
%run -i src/data_preparation.py
%run -i src/model_configuration.py
%run -i src/model_training.py
```

For image classification by simple DNN:
```python
%run -i src/image_preparation.py
%run -i src/model_configuration.py
%run -i src/model/utils.py
mymodel = MyTrainingClass(model, loss_fn, optimizer)
mymodel.set_loaders(train_loader, val_loader)
mymodel.set_tensorboard("image_clfier_relu")
mymodel.train(n_epochs=50)
```

# Issues

Iss-01: 
Using composer of Normalize(mean=0.5, std=0.5)
-> 图像像素值为-1和1
-> cnn will always predict the same class

Using composer of Normalize(mean=0.2, std=0.5)
-> 图像像素值为(-1,1)区间
-> cnn will misclassify few samples only

Seems like this is related to -1/1 imput value of images
