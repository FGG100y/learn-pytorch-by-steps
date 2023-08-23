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
