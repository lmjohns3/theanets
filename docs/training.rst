================
Training a Model
================

.. _training-gradient-methods:

Gradient-Based Training Methods
===============================

Stochastic Gradient Descent
---------------------------
sgd: Stochastic Gradient Descent
  --learning-rate
  --momentum

nag: Nesterov's Accelerated Gradient
  --learning-rate
  --momentum

rprop: Resilient Backpropagation
  --learning-rate (sets initial learning rate)
  --rprop-increase, --rprop-decrease
  --rprop-min-step, --rprop-max-step

rmsprop: RMS-scaled Backpropagation
  --learning-rate
  --momentum
  --rms-halflife

adadelta: ADADELTA
  --rms-halflife

bfgs, cg, dogleg, newton-cg, trust-ncg
  These use the implementations in scipy.optimize.minimize.

Second-Order Gradient Descent
-----------------------------
hf: Hessian-Free
  --cg-batches
  --initial-lambda
  --global-backtracking
  --preconditioner

.. _training-other-methods:

Other Training Methods
======================

Sampling from data
------------------

sample: Set model parameters to training data samples

Layerwise pretraining
---------------------

layerwise: Greedy supervised layerwise pre-training
  This trainer applies RmsProp to each layer sequentially.

pretrain: Greedy unsupervised layerwise pre-training.
  This trainer applies RmsProp to a tied-weights "shadow" autoencoder using an
  unlabeled dataset, and then transfers the learned autoencoder weights to the
  model being trained.

.. _training-providing-data:

Providing Data
==============

.. _training-using-arrays:

Using arrays
------------

A fairly typical use case for training a neural network in Python is to
construct a ``numpy`` array containing the data you have.

.. _training-using-callables:

Using callables
---------------

You can provide a callable for a dataset. This callable must take no arguments
and must return a ``numpy`` array of the proper shape for your model.

For example, this code defines a ``batch()`` helper that chooses a random
dataset and a random offset for each batch::

  SOURCES = 'foo.npy', 'bar.npy', 'baz.npy'
  BATCH_SIZE = 64

  def batch():
      X = np.load(np.random.choice(SOURCES), mmap_mode='r')
      i = np.random.randint(len(X))
      return X[i:i+BATCH_SIZE]

  # ...

  exp.train(batch)

If you need to maintain more state than is reasonable from a single closure, you
can also encapsulate the callable inside a class. Just make sure instances of
the class are callable by defining the ``__call__`` method::

  class Loader:
      def __init__(sources=('foo.npy', 'bar.npy', 'baz.npy'), batch_size=64):
          self.sources = sources
          self.batch_size = batch_size
          self.src = -1
          self.idx = 0
          self.X = ()

      def __call__(self):
          if self.idx + self.batch_size > len(self.X):
              self.idx = 0
              self.src = (self.src + 1) % len(self.sources)
              self.X = np.load(self.sources[self.src], mmap_mode='r')
          try:
              return self.X[self.idx:self.idx+self.batch_size]
          finally:
              self.idx += self.batch_size

  # ...

  exp.train(Loader())

.. _training:

Training
========

.. _training-iteration:

Training as iteration
---------------------

The :func:`Experiment.train() <theanets.main.Experiment.train>` method is
actually just a thin wrapper over the underlying :func:`Experiment.itertrain()
<theanets.main.Experiment.itertrain>` method, which you can use directly if you
want to do something special during training::

  for train, valid in exp.itertrain(train_dataset, valid_dataset, **kwargs):
      print('training loss:', train['loss'])
      print('most recent validation loss:', valid['loss'])

Trainers yield a dictionary after each training iteration. The keys and values
in each dictionary give the costs and monitors that are computed during
training, which will vary depending on the model being trained. However, there
will always be a ``'loss'`` key that gives the value of the loss function being
optimized. Many types of models have an ``'err'`` key that gives the values of
the unregularized error (e.g., the mean squared error for regressors). For
classifier models, the dictionary will also have an ``'acc'`` key, which
contains the percent accuracy of the classifier model.

.. _training-saving-progress:

Saving progress
---------------

The :class:`Experiment <theanets.main.Experiment>` class can snapshot your model
automatically during training. When you call :func:`Experiment.train()
<theanets.main.Experiment.train>`, you can provide the following keyword
arguments:

- ``save_progress``: This should be a string containing a filename where the
  model should be saved.

- ``save_every``: This should be a numeric value specifying how often the model
  should be saved during training. If this value is positive, it specifies the
  number of training iterations between checkpoints; if it is negative, it
  specifies the number of minutes that are allowed to elapse between
  checkpoints.

If you provide a ``save_progress`` argument when you construct your experiment,
and a model exists in the given snapshot file, then that model will be loaded
from disk.

You can also save and load models manually by calling :func:`Experiment.save()
<theanets.main.Experiment.save>` and :func:`Experiment.load()
<theanets.main.Experiment.load>`, respectively.
