# how this scripts work

1. linear_regression_in_numpy.py, linear_regression_in_pytorch.py

```sh
python3 linear_regression_in_*.py
```

2. scripts in src/

These scripts should be viewed as jupyter cell code (which need an shared
namespace of all the functions/modules. I mean that:

a. Get into jupyterlab, jupyter notebook, or IPython (if you like shell)
b. In IPython console, run these commands in order (-i makes shared namespcace):
```python
%run -i src/data_preparation.py
%run -i src/model_configuration.py
%run -i src/model_training.py
```
