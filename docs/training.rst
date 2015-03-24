================
Training a Model
================

When most neural network models are created, their parameters are set to small
random values. These values are not particularly well-suited to perform most
tasks, so some sort of training process is needed to optimize the parameters for
the task that the network should perform.

The neural networks research literature is filled with exciting advances in
training algorithms for neural networks. In ``theanets`` there are several
algorithm implementations available; each one has different performance
characteristics and might be better or worse suited for a particular model or
task.

Specifying a Trainer
====================

The easiest way train a model with ``theanets`` is to use the :class:`Experiment
<theanets.main.Experiment>` class::

  exp = theanets.Experiment(theanets.Classifier, layers=(10, 5, 2))
  exp.train(training_data,
            validation_data,
            optimize='nag',
            learning_rate=0.01)

Here, a classifier model is being trained using Nesterov's accelerated gradient,
with a learning rate of 0.01. The training and validation datasets must be
provided to any of the available training algorithms. The algorithm itself is
selected using the ``optimize`` keyword argument, and any other keyword
arguments provided to ``train()`` are passed to the algorithm implementation.

Multiple calls to ``train()`` are possible and can be used to implement things
like custom annealing schedules::

  exp = theanets.Experiment(theanets.Classifier, layers=(10, 5, 2))
  exp.train(training_data,
            validation_data,
            optimize='nag',
            learning_rate=0.01,
            momentum=0.9)
  exp.train(training_data,
            validation_data,
            optimize='nag',
            learning_rate=0.001,
            momentum=0.99)
  exp.train(training_data,
            validation_data,
            optimize='sgd',
            learning_rate=0.0001,
            momentum=0.999)

The available training methods are described below, followed by some details on
additional functionality available for training models.

.. _training-gradient-methods:

Gradient-Based Methods
======================

Stochastic Gradient Descent
---------------------------

:class:`sgd <theanets.trainer.SGD>`

  --learning-rate
  --momentum

Stochastic gradient descent is the simplest and probably most widely used
training method for neural network models. It adjusts parameters at each
training iteration by taking a step in the direction of the gradient of the loss
function with respect to the parameter. These steps can additionally be combined
with a momentum term that helps to incorporate information from previous time
steps, thus computing some sort of approximate information about second-order
derivatives in the updates.

:class:`nag <theanets.trainer.NAG>`

  --learning-rate
  --momentum

Nesterov's accelerated gradient extends vanilla stochastic gradient by computing
the gradient at a different point in parameter space than the default. By
altering the computation slightly in this way, it is thought that NAG helps to
avoid overstepping during training.

:class:`rprop <theanets.trainer.Rprop>`

  --learning-rate (sets initial learning rate)
  --rprop-increase, --rprop-decrease
  --rprop-min-step, --rprop-max-step

Resilient backpropagation is a technique for making gradient-based updates that
incorporates the most recent update history into the step size. For each
parameter, a step size is maintained during training. Whenever the signs of the
gradients agree on consecutive training iterations, the step size increases;
whenever the signs of consecutive gradients disagree, the step size decreases.

:class:`rmsprop <theanets.trainer.RmsProp>`

  --learning-rate
  --momentum
  --rms-halflife

RmsProp takes a step in the direction of the current gradient, but it scales the
step inversely by the "recent average" gradient value. The recent history is
maintained using an exponentially weighted moving average of gradient magnitudes.

:class:`adadelta <theanets.trainer.ADADELTA>`

  --rms-halflife

ADADELTA is a training method that incorporates an exponentially weighted moving
average of both the recent gradient values as well as the recent update sizes
for each parameter. It is very similar to RmsProp.

:class:`esgd <theanets.trainer.ESGD>`

  --learning-rate
  --momentum
  --rms-halflife

Equilibrated SGD takes a step in the direction of the current gradient, but it
scales the step inversely by a preconditioner, in this case an estimate of the
absolute value of the diagonal of the Hessian. The preconditioner is maintained
using an exponentially weighted moving average of :math:`Hv` (Hessian-vector)
products.

bfgs, cg, dogleg, newton-cg, trust-ncg (:class:`theanets.trainer.Scipy`)

These trainers use the implementations in `scipy.optimize.minimize`_: the loss
function and its gradient are computed by ``theanets``, possibly on the GPU, but
the resulting values are always processed on the CPU using the ``scipy``
routines. Depending on your model and task, these trainers might be very slow.

.. _scipy.optimize.minimize: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Second-Order Gradient Descent
-----------------------------

:class:`hf <theanets.trainer.HF>`

  --cg-batches
  --initial-lambda
  --global-backtracking
  --preconditioner

This trainer takes gradient steps that take into account both the first *and*
the second derivative of the loss function with respect to the model parameters.
It is quite slow compared with other training methods, but can sometimes yield
much better performance.

This trainer does not work with Python3. Also, currently this trainer is not
functioning properly with Python2; see
https://github.com/lmjohns3/theanets/issues/62 for details.

.. _training-other-methods:

Other Training Methods
======================

Sampling from data
------------------

:class:`sample <theanets.trainer.Sample>`

This trainer sets model parameters directly to samples drawn from the training
data. This is a very fast "training" algorithm since all updates take place at
once; however, often features derived directly from the training data require
further tuning to perform well.

Layerwise pretraining
---------------------

:class:`layerwise <theanets.trainer.SupervisedPretrainer>`

Greedy supervised layerwise pre-training: This trainer applies RmsProp to each
layer sequentially.

:class:`pretrain <theanets.trainer.UnsupervisedPretrainer>`

Greedy unsupervised layerwise pre-training: This trainer applies RmsProp to a
tied-weights "shadow" autoencoder using an unlabeled dataset, and then transfers
the learned autoencoder weights to the model being trained.

.. _training-providing-data:

Providing Data
==============

One of the areas in ``theanets`` that consistently requires the most work is
assembling data to use when training your model.

.. _training-using-arrays:

Using arrays
------------

A fairly typical use case for training a neural network in Python is to
construct a ``numpy`` array containing the data you have::

  dataset = np.load(filename)

  exp = theanets.Experiment()
  exp.train(dataset)

Sometimes the data available for training a network model exceeds the available
resources (e.g., memory) on the computer at hand. There are several ways of
handling this type of situation. If your data are already in a ``numpy`` array
stored on disk, you might want to try loading the array using ``mmap``::

  dataset = np.load(filename, mmap_mode='r')

  exp = theanets.Experiment()
  exp.train(dataset)

Alternatively, you might want to load just part of the data and train on that,
then load another part and train on it::

  exp = theanets.Experiment()
  for filename in data_files:
    dataset = np.load(filename)
    exp.train(dataset)

Finally, you can potentially handle large datasets by using a callable to
provide data to the training algorithm.

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

.. _training-specifying-regularizers:

Specifying Regularizers
=======================

One heuristic that can prevent models from overtraining on small datasets is
based on the observation that "good" parameter values are typically small: large
parameter values often indicate overfitting.

One way to encourage a model to use small parameter values is to assume that the
parameter values are sampled from a posterior distribution over parameters,
conditioned on observed data. In this way of thinking about parameters, we can
manipulate the prior distribution of the parameter values to express our
knowledge as modelers of the problem at hand.

If you want to set up a more sophisticated model like a classifier with sparse
hidden representations, you can add regularization hyperparameters when you
create your experiment::

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(784, 1000, 784),
  )
  exp.train(dataset, hidden_l1=0.1)

Here we've specified that our model has a single, overcomplete hidden layer, and
then when we train it, we specify that the activity of the hidden units in the
network will be penalized with a 0.1 coefficient.

Decay
-----

In "weight decay," we assume that parameters are drawn from a zero-mean Gaussian
distribution with an isotropic, modeler-specified standard deviation. In terms
of loss functions, this equates to adding a term to the loss function that
computes the :math:`L_2` norm of the parameter values in the model:

.. math::
   J(\cdot) = \dots + \frac{\lambda}{2} \| \theta \|_2^2

If the loss :math:`J(\cdot)` represents some approximation to the log-posterior
distribution of the model parameters given the data

.. math::
   J(\cdot) = \log p(\theta|x) \propto \dots + \frac{\lambda}{2} \| \theta \|_2^2

then the term with the :math:`L_2` norm on the parameters is like an unscaled
Gaussian distribution.

Sparsity
--------

Sparse models have been shown to capture regularities seen in the mammalian
visual cortex [Ols94]_. In addition, sparse models in machine learning are often
more performant than "dense" models (i.e., models without restriction on the
hidden representation) [Lee08]_. Furthermore, sparse models tend to yield latent
representations that are more interpretable to humans than dense models
[Tib96]_.

.. _training-training:

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
