.. image:: https://travis-ci.org/lmjohns3/theanets.svg?branch=master
.. image:: https://coveralls.io/repos/lmjohns3/theanets/badge.svg?branch=master
   :target: https://coveralls.io/r/lmjohns3/theanets?branch=master

============
``THEANETS``
============

The ``theanets`` package is a deep learning and neural network toolkit. It is
written in Python to interoperate with excellent tools like ``numpy`` and
``scikit-learn``, and it uses Theano_ to take advantage of your GPU. The package
aims to provide:

- a simple API for building and training common types of neural network models;
- thorough documentation;
- easy-to-read code;
- and, under the hood, a fully expressive graph computation framework.

The package strives to "make the easy things easy and the difficult things
possible." Please try it out, and let us know what you think!

.. _Theano: http://deeplearning.net/software/theano/

Installation
============

Install the latest published code using pip::

    pip install theanets

Or download the current source and run it from there::

    git clone http://github.com/lmjohns3/theanets
    cd theanets
    python setup.py develop

Quick Start: Classification
===========================

Suppose you want to create a classifier and train it on some 100-dimensional
data points that you've classified into 10 categories. No problem! With just a
few lines you can (a) provide some data, (b) build and (c) train a model,
and (d) evaluate the model::

  import theanets
  from sklearn.datasets import make_classification
  from sklearn.metrics import confusion_matrix

  # Create a classification dataset.
  X, y = make_classification(n_samples=3000, n_features=100, n_classes=10)
  X = X.astype('f')
  y = y.astype('i')
  cut = int(len(X) * 0.8)  # training / validation split
  train = X[:cut], y[:cut]
  valid = X[cut:], y[cut:]

  # Build a classifier model with 100 inputs and 10 outputs.
  exp = theanets.Experiment(theanets.Classifier, layers=(100, 10))

  # Train the model using SGD with momentum.
  exp.train(train, valid, algo='sgd', learning_rate=1e-4, momentum=0.9)

  # Show confusion matrices on the training/validation splits.
  for label, (X, y) in (('training:', train), ('validation:', valid)):
      print(label)
      print(confusion_matrix(y, exp.network.predict(X)))

Layers
------

The model above is quite simplistic! Make it a bit more sophisticated by adding
a hidden layer::

  exp = theanets.Experiment(theanets.Classifier, (100, 1000, 10))

In fact, you can just as easily create 3 (or any number of) hidden layers::

  exp = theanets.Experiment(
      theanets.Classifier,
      (100, 1000, 1000, 1000, 10))

By default, hidden layers use the logistic sigmoid transfer function. By passing
a tuple instead of just an integer, you can change some of these layers to use
different activations_::

  maxout = (1000, 'maxout:4')  # maxout with 4 pieces.
  exp = theanets.Experiment(
      theanets.Classifier,
      (100, 1000, maxout, (1000, 'relu'), 10))

.. _activations: http://theanets.readthedocs.org/en/latest/reference.html#module-theanets.activations

By passing a dictionary instead, you can specify even more attributes of each
layer_, like how its parameters are initialized::

  # Sparsely-initialized layer with large nonzero weights.
  foo = dict(name='foo', size=1000, std=1, sparsity=0.9)
  exp = theanets.Experiment(
      theanets.Classifier,
      (100, foo, (1000, 'maxout:4'), (1000, 'relu'), 10))

.. _layer: http://theanets.readthedocs.org/en/latest/reference.html#module-theanets.layers.base

Specifying layers is the heart of building models in ``theanets``. Read more
about this in `Creating a Model`_.

.. _Creating a Model: http://theanets.readthedocs.org/en/latest/creating.html

Regularization
--------------

Adding regularizers is easy, too! Just pass them to the training method. For
instance, you can train up a sparse classification model with weight decay::

  # Penalize hidden-unit activity (L1 norm) and weights (L2 norm).
  exp.train(train, valid, hidden_l1=0.001, weight_l2=0.001)

In ``theanets`` dropout is treated as a regularizer and can be set on many
layers at once::

  exp.train(train, valid, hidden_dropout=0.5)

or just on a specific layer::

  exp.train(train, valid, dropout={'foo:out': 0.5})

Similarly, you can add Gaussian noise to any of the layers (here, just to the
input layer)::

  exp.train(train, valid, input_noise=0.3)

You can make as many successive calls to train() as you like. Each call can
include different training algorithms::

  exp.train(train, valid, algo='rmsprop')
  exp.train(train, valid, algo='nag')

different learning hyperparameters::

  exp.train(train, valid, algo='rmsprop', learning_rate=0.1)
  exp.train(train, valid, algo='rmsprop', learning_rate=0.01)

and different regularization hyperparameters::

  exp.train(train, valid, input_noise=0.7)
  exp.train(train, valid, input_noise=0.3)

Training models is a bit more art than science, but ``theanets`` tries to make
it easy to evaluate different training approaches. Read more about this in
`Training a Model`_.

.. _Training a Model: http://theanets.readthedocs.org/en/latest/training.html

Quick Start: Recurrent Models
=============================



More Information
================

Source: https://github.com/lmjohns3/theanets

Documentation: http://theanets.readthedocs.org

Mailing list: https://groups.google.com/forum/#!forum/theanets
