================
Training a Model
================

When most neural network models are created, their parameters are set to small
random values. These values are not particularly well-suited to perform most
tasks, so some sort of training process is needed to optimize the parameters for
the task that the network should perform.

The neural networks research literature is filled with exciting advances in
optimization algorithms for neural networks. In ``theanets`` several optimizers
are available; each one has different performance characteristics and might be
better or worse suited for a particular model or task.

To train a network, you must first specify a trainer and then provide some data
to the trainer. You can also save the model periodically during training.

Specifying a Trainer
====================

The easiest way train a model with ``theanets`` is to use the :class:`Experiment
<theanets.main.Experiment>` class::

  exp = theanets.Experiment(theanets.Classifier, layers=(10, 5, 2))
  exp.train(training_data,
            validation_data,
            algorithm='nag',
            learning_rate=0.01,
            momentum=0.9)

Here, a classifier model is being trained using `Nesterov's accelerated
gradient`_, with a learning rate of 0.01 and momentum of 0.9. The training and
validation datasets must be provided to any of the available training
algorithms. The algorithm itself is selected using the ``algorithm`` keyword
argument, and any other keyword arguments provided to ``train()`` are passed to
the algorithm implementation.

Multiple calls to ``train()`` are possible and can be used to implement things
like custom annealing schedules (e.g., the "newbob" training strategy)::

  exp = theanets.Experiment(theanets.Classifier, layers=(10, 5, 2))

  for e in (-2, -3, -4):
    exp.train(training_data,
              validation_data,
              algorithm='nag',
              learning_rate=10 ** e,
              momentum=1 - 10 ** (e + 1))

  exp.train(training_data,
            validation_data,
            algorithm='rmsprop',
            learning_rate=0.0001,
            momentum=0.9)

The available training methods are described below, followed by some details on
additional functionality available for training models.

.. _training-available-trainers:

Available Trainers
==================

The most common method for training a neural network model is to use a
stochastic gradient-based optimizer. In ``theanets`` many of these algorithms
are available by interfacing with the ``downhill`` package:

- ``sgd``: `Stochastic gradient descent`_
- ``nag``: `Nesterov's accelerated gradient`_
- ``rprop``: `Resilient backpropagation`_
- ``rmsprop``: RMSProp_
- ``adadelta``: ADADELTA_
- ``esgd``: `Equilibrated SGD`_
- ``adam``: Adam_

.. _Stochastic gradient descent: http://downhill.readthedocs.org/en/stable/generated/downhill.first_order.SGD.html
.. _Nesterov's accelerated gradient: http://downhill.readthedocs.org/en/stable/generated/downhill.first_order.NAG.html
.. _Resilient backpropagation: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.RProp.html
.. _RMSProp: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.RMSProp.html
.. _ADADELTA: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ADADELTA.html
.. _Equilibrated SGD: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ESGD.html
.. _Adam: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.Adam.html

In addition to the optimization algorithms provided by ``downhill``,
``theanets`` defines a few algorithms that are more specific to neural networks.
These trainers tend to take advantage of the layered structure of the loss
function for a network.

- ``sample``: :class:`Sample trainer <theanets.trainer.Sample>`

This trainer sets model parameters directly to samples drawn from the training
data. This is a very fast "training" algorithm since all updates take place at
once; however, often features derived directly from the training data require
further tuning to perform well.

- ``layerwise``: :class:`Layerwise (supervised) pretrainer <theanets.trainer.SupervisedPretrainer>`

Greedy supervised layerwise pre-training: This trainer applies RMSProp to each
layer sequentially.

- ``pretrain``: :class:`Unsupervised pretrainer <theanets.trainer.UnsupervisedPretrainer>`

Greedy unsupervised layerwise pre-training: This trainer applies RMSProp to a
tied-weights "shadow" autoencoder using an unlabeled dataset, and then transfers
the learned autoencoder weights to the model being trained.

.. _training-providing-data:

Providing Data
==============

One of the areas in ``theanets`` that consistently requires the most work is
assembling data to use when training your model.

.. _training-using-arrays:

Using Arrays
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

Using Callables
---------------

Instead of an array of data, you can provide a callable for a dataset. This
callable must take no arguments and must return one or more ``numpy`` arrays of
the proper shape for your model.

During training, the callable will be invoked every time the training algorithm
requires a batch of training (or validation) data. Therefore, your callable
should return at least one array containing a batch of data; if your model
requires multiple arrays per batch (e.g., if you are training a
:class:`classification <theanets.feedforward.Classifier>` or :class:`regression
<theanets.feedforward.Regressor>` model), then your callable should return a
list containing the correct number of arrays (e.g., a training array and the
corresponding labels).

For example, this code defines a ``batch()`` helper that could be used when
training a plain :class:`autoencoder <theanets.feedforward.Autoencoder>` model.
The callable chooses a random dataset and a random offset for each batch::

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
the class are callable by defining the ``__call__`` method. For example, this
class loads data from a series of ``numpy`` arrays on disk, but only loads one
of the on-disk arrays into memory at a given time::

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

Thanks to Python's flexibility in making classes callable, there are almost
limitless possibilities for using callables to interface with the training
process.

.. _training-specifying-hyperparameters:

Specifying Hyperparameters
==========================

A training algorithm typically relies on a small number of "hyperparameters" to
define how it interprets loss and gradient information from the model during
training. For example, many stochastic gradient-based optimization algorithms
rely on a learning rate parameter to specify the scale of the parameter updates
to apply.

In ``theanets`` these hyperparameters are specified separately as keyword
arguments during each call to ``train()``. Although some training approaches
offer specialized hyperparameters, here we'll cover a few of the hyperparameters
that are common to most algorithms.

Learning Rate
-------------

The most basic stochastic gradient optimization method makes small parameter
updates based on the local gradient of the loss at each step in the optimization
procedure. Intuitively, parameters in a model are updated by subtracting a small
portion of the local derivative from the current parameter value.
Mathematically, this is written as:

.. math::

   \theta_{t+1} = \theta_t - \alpha \left. \frac{\partial\mathcal{L}}{\partial\theta} \right|_{\theta_t}

where :math:`\mathcal{L}` is the loss function being optimized, :math:`\theta`
is the value of a parameter in the model at optimization step :math:`t`,
:math:`\alpha` is the learning rate, and
:math:`\frac{\partial\mathcal{L}}{\partial\theta}` (also often written
:math:`\nabla_{\theta_t}\mathcal{L}`) is the partial derivative of the loss with
respect to the parameters, evaluated at the current value of those parameters.

The learning rate :math:`\alpha` specifies the scale of these parameter updates
with respect to the magnitude of the gradient. Almost all stochastic optimizers
use a fixed learning rate parameter.

In ``theanets``, the learning rate is passed as a keyword argument to
``train()``::

  exp.train(data, learning_rate=0.1)

Often the learning rate is set to a very small value---many approaches seem to
start with values around 1e-4. If the learning rate is too large, the
optimization procedure might "bounce around" in the loss landscape because the
parameter steps are too large. If the learning rate is too small, the
optimization procedure might not make progress quickly enough to make training
practical.

Momentum
--------

Momentum is a common technique in stochastic gradient optimization algorithms
that seems to accelerate the optimization process in most cases. Intuitively,
momentum maintains a "velocity" of the most recent parameter steps and combines
these recent individual steps together when making a parameter update. By
combining individual steps, momentum tends to "smooth out" any outliers in the
update process. Mathematically, this is written:

.. math::

   \begin{eqnarray*}
   \nu_{t+1} &=& \mu \nu_t - \alpha \left. \frac{\partial\mathcal{L}}{\partial\theta} \right|_{\theta_t} \\
   \theta_{t+1} &=& \theta_t + \nu_{t+1}
   \end{eqnarray*}

where the symbols are the same as the description of vanilla SGD above,
:math:`\nu` describes the "velocity" of parameter :math:`\theta`, and
:math:`\mu` is the momentum hyperparameter. The gradient computations using
momentum are exactly the same as when not using momentum; the only difference is
the accumulation of recent updates in the "velocity."

In ``theanets``, the momentum value is passed as a keyword argument to
``train()``::

  exp.train(data, momentum=0.9)

Typically momentum is set to a value in :math:`[0, 1)`---when set to 0, momentum
is disabled, and when set to values near 1, the momentum is very high, requiring
several consecutive parameter updates in the same direction to change the
parameter velocity. Often it is useful to set the momentum to a surprisingly
large value, sometimes even to values greater than 0.9. Such values can be
especially effective with a relatively small learning rate. If the momentum is
set too low, then parameter updates will be more noisy and optimization might
take longer to converge, but if the momentum is set too high, the optimization
process might diverge entirely.

Early Stopping
--------------

When you make a call to ``train()`` (or ``itertrain()``), ``theanets`` begins an
optimization procedure.

continue to iterate as long as the training procedure you're using doesn't run
out of patience. So the 50 iterations you're seeing might vary depending on the
model, your dataset, and your training algorithm & parameters. (E.g., the
"sample" trainer only produces one result, because sampling from the training
dataset just happens once, but the SGD-based trainers will run for multiple
iterations.)

For each iteration produced by itertrain using a SGD-based algorithm, the
trainer applies "train_batches" gradient updates to the model. Each of these
batches contains "batch_size" training examples and computes a single gradient
update. After "train_batches" have been processed, the training dataset is
shuffled, so that subsequent iterations might see the same set of batches, but
not in the same order.

The validation dataset is run through the model to test convergence every
"validate_every" iterations. If there is no progress for "patience" of these
validations, then the training algorithm halts and returns.

In theanets, the patience is the number of failed validation attempts
that we're willing to tolerate before seeing any progress. So theanets
will make (patience * validate_every) training updates, checking
(patience) times for improvement before deciding that training should
halt.

In some other tools, the patience is the number of training updates
that we're willing to wait before seeing any progress; these tools
will make (patience) training updates, checking (patience /
validate_every) times for improvement before deciding that training
should halt. With this definition, you do want to make sure the
validation frequency is smaller than half the patience, to have a good
chance of seeing progress before halting.

Gradient Clipping
-----------------

.. _training-specifying-regularizers:

Specifying Regularizers
=======================

The goal of training a model is to minimize the loss function by making
adjustments to the model parameters. In most practical applications, the loss is
not known a priori, but an estimate of it is computed using a set of data (the
"training data") that has been gathered from the problem being modeled.

If a model has many parameters compared with the size of the training dataset,
then many machine learning models exhibit a phenomenon called *overfitting*: the
model may learn to predict the training data with no measurable error, but then
if it is applied to a new dataset, it makes lots of mistakes. In such a case,
the model has essentially memorized the training data at the cost of not being
able to *generalize* to new and unseen, yet similar, datasets. The risk of
overfitting usually increases with the size of the model (as measured by the
number of parameters) and decreases with the size of the training dataset.

Another heuristic that can prevent models from overfitting on small datasets is
based on the observation that "good" parameter values in most models are
typically small: large parameter values often indicate overfitting.

One way to encourage a model to use small parameter values is to assume that the
parameter values are sampled from some prior distribution, rather than assuming
that all parameter values in the model are equally likely. In this way of
thinking about parameters, we can manipulate the prior distribution of the
parameter values to express our knowledge as modelers of the problem at hand.

In ``theanets``, regularization hyperparameters are provided when you train your
model::

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(784, 1000, 784),
  )
  exp.train(dataset, hidden_l1=0.1)

Here we've specified that our model has a single, overcomplete hidden layer, and
then when we train it, we specify that the activity of the hidden units in the
network will be penalized with a 0.1 coefficient. The rest of this section
details the built-in regularizers that are available in ``theanets``.

Input Regularization
--------------------

One way of regularizing a model to prevent overfitting is to add noise to the
data during training. While noise could be added in the training batches,
``theanets`` provides two types of input noise regularizers: Gaussian noise and
dropouts.

In one method, zero-mean Gaussian noise is added to the input data; this is
specified during training using the ``input_noise`` keyword argument::

  exp.train(dataset, input_noise=0.1)

The value of the argument specifies the standard deviation of the noise.

In the other input regularization method, some of the inputs are randomly set to
zero during training (this is sometimes called "dropout" or "masking noise").
This type of input noise is specified using the ``input_dropout`` keyword
argument::

  exp.train(dataset, input_dropout=0.3)

The value of the argument specifies the fraction of values in each input vector
that are randomly set to zero.

Decay
-----

In "weight decay," we assume that parameters are drawn from a zero-mean Gaussian
distribution with an isotropic, modeler-specified standard deviation. In terms
of loss functions, this equates to adding a term to the loss function that
computes the :math:`L_2` norm of the parameter values in the model:

.. math::
   \mathcal{L}(\cdot) = \dots + \frac{\lambda}{2} \| \theta \|_2^2

If the loss :math:`\mathcal{L}(\cdot)` represents some approximation to the
log-posterior distribution of the model parameters given the data

.. math::
   \mathcal{L}(\cdot) = \log p(\theta|x) \propto \dots + \frac{\lambda}{2} \| \theta \|_2^2

then the term with the :math:`L_2` norm on the parameters is like an unscaled
Gaussian distribution.

This type of regularization is specified using the ``weight_l2`` keyword
argument during training::

  exp.train(dataset, weight_l2=1e-4)

The value of the argument is the strength of the regularizer in the loss for the
model. Smaller values create less pressure for small model weights.

Sparsity
--------

Sparse models have been shown to capture regularities seen in the mammalian
visual cortex [Ols94]_. In addition, sparse models in machine learning are often
more performant than "dense" models (i.e., models without restriction on the
hidden representation) [Lee08]_. Furthermore, sparse models tend to yield latent
representations that are more interpretable to humans than dense models
[Tib96]_.

There are two main types of sparsity provided with ``theanets``: parameter
sparsity and representation sparsity.

The first type of sparse regularizer is just like weight decay, but instead of
assuming that weights are drawn from a Gaussian distribution, here we assume
that weights in the model are drawn from a distribution with a taller peak at
zero, like a Laplace distribution. In terms of loss function, this regularizer
adds a term with an :math:`L_1` norm to the model:

.. math::
   \mathcal{L}(\cdot) = \dots + \lambda \| \theta \|_1

If the loss :math:`\mathcal{L}(\cdot)` represents some approximation to the
log-posterior distribution of the model parameters given the data

.. math::
   \mathcal{L}(\cdot) = \log p(\theta|x) \propto \dots + \lambda \| \theta \|_1

then this term is like an unscaled Laplace distribution. In practice, this
regularizer encourages many of the model parameters to be zeros.

In ``theanets``, this sparse parameter regularization is specified using the
``weight_l1`` keyword argument during training::

  exp.train(dataset, weight_l1=1e-4)

The value of the argument is the strength of the regularizer in the loss for the
model. Smaller values create less pressure for sparse model weights.

The second type of sparsity regularization puts pressure on the model to develop
hidden representations that use as few nonzero values as possible. In this type
of regularization, the model weights are penalized indirectly, since the hidden
representation (i.e., the values of the hidden layer neurons in the network) are
functions of both the model weights and the input data.

Sparse hidden activations have shown much promise in computational neural
networks. In ``theanets`` this type of regularization is specified using the
``hidden_l1`` keyword argument during training::

  exp.train(dataset, hidden_l1=0.1)

The value of the argument is the strength of the regularizer in the loss for the
model. Smaller values create less pressure for sparse hidden representations.

.. _training-training:

Training
========

.. _training-iteration:

Training as Iteration
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

Saving Progress
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
