

## Setup Pipenv

```
pipenv --python 3.6
```


## Install mxnet-cu102

```
pipenv run pip install mxnet-cu102mkl==1.6.0 -f https://dist.mxnet.io/python
```

## Run
```
$> pipenv shell
$> python

>>> import sys
>>> sys.path.append('src')
>>> import trainer
>>> trainer.train()
```
