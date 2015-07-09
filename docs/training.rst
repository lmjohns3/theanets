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

The easiest way train a model with ``theanets`` is to invoke the :func:`train()
<theanets.graph.Network.train>` method::

  net = theanets.Classifier(layers=[10, 5, 2])
  net.train(training_data,
            validation_data,
            algo='nag',
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

  net = theanets.Classifier(layers=[10, 5, 2])

  for e in (-2, -3, -4):
      net.train(training_data,
                validation_data,
                algo='nag',
                learning_rate=10 ** e,
                momentum=1 - 10 ** (e + 1))

  net.train(training_data,
            validation_data,
            algo='rmsprop',
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

- ``sample``: :class:`Sample trainer <theanets.trainer.SampleTrainer>`

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

To train a model in ``theanets``, you will need to provide a set of data that
can be used to compute the value of the loss function and its derivatives. Data
can be passed to the trainer using either arrays_ or callables_; the
``downhill`` documentation describes how this works.

.. _arrays: http://downhill.rtfd.org/en/stable/guide.html#data-using-arrays
.. _callables: http://downhill.rtfd.org/en/stable/guide.html#data-using-callables

.. _training-specifying-regularizers:

Specifying Regularizers
=======================

The goal of training a neural network model is to minimize the loss function by
making adjustments to the model parameters. In most practical applications, the
loss is not known a priori, but an estimate of it is computed using a set of
data (the "training data") that has been gathered from the problem being
modeled.

If a model has many parameters compared with the size of the training dataset,
then many machine learning models exhibit a phenomenon called *overfitting*: the
model may learn to predict the training data with no measurable error, but then
if it is applied to a new dataset, it makes lots of mistakes. In such a case,
the model has essentially memorized the training data at the cost of not being
able to *generalize* to new and unseen, yet similar, datasets. The risk of
overfitting usually increases with the size of the model and decreases with the
size of the training dataset.

A heuristic that can prevent models from overfitting on small datasets is based
on the observation that "good" parameter values in most models are typically
small: large parameter values often indicate overfitting.

One way to encourage a model to use small parameter values is to assume that the
parameter values are sampled from some prior distribution, rather than assuming
that all parameter values in the model are equally likely. In this way of
thinking about parameters, we can manipulate the prior distribution of the
parameter values to express our knowledge as modelers of the problem at hand.

In ``theanets``, regularization hyperparameters are provided when you train your
model::

  net = theanets.Classifier(layers=[784, 1000, 784])
  net.train(..., hidden_l1=0.1)

Here we've specified that our model has a single, overcomplete hidden layer, and
then when we train it, we specify that the activity of the hidden units in the
network will be penalized with a 0.1 coefficient. The rest of this section
details the built-in regularizers that are available in ``theanets``.

Decay
-----

Using "weight decay," we assume that parameters in a model are drawn from a
zero-mean Gaussian distribution with an isotropic, modeler-specified standard
deviation. In terms of loss functions, this equates to adding a term to the loss
function that computes the :math:`L_2` norm of the parameter values in the
model:

.. math::
   \mathcal{L}(\cdot) = \dots + \lambda \| \theta \|_2^2

If the loss :math:`\mathcal{L}(\cdot)` represents some approximation to the
log-posterior distribution of the model parameters given the data

.. math::
   \mathcal{L}(\cdot) = \log p(\theta|x) \propto \dots + \lambda \| \theta \|_2^2

then the term with the :math:`L_2` norm on the parameters is like an unscaled
Gaussian distribution.

This type of regularization is specified using the ``weight_l2`` keyword
argument during training::

  net.train(..., weight_l2=1e-4)

The value of the argument is the strength of the regularizer in the loss for the
model. Larger values create more pressure for small model weights.

Sparsity
--------

Sparse models have been shown to capture regularities seen in the mammalian
visual cortex. In addition, sparse models in machine learning are often more
performant than "dense" models (i.e., models without restriction on the hidden
representation). Furthermore, sparse models tend to yield latent representations
that are easier for humans to interpret than dense models.

There are two main types of sparsity regularizers provided with ``theanets``:
parameter sparsity and representation sparsity.

The first type of sparse regularizer is just like weight decay, but instead of
assuming that weights are drawn from a Gaussian distribution, here we assume
that weights in the model are drawn from a distribution with a taller peak at
zero and heavier tails, like a Laplace distribution. In terms of loss function,
this regularizer adds a term with an :math:`L_1` norm to the model:

.. math::
   \mathcal{L}(\cdot) = \dots + \lambda \| \theta \|_1

If the loss :math:`\mathcal{L}(\cdot)` represents some approximation to the
log-posterior distribution of the model parameters given the data

.. math::
   \mathcal{L}(\cdot) = \log p(\theta|x) \propto \dots + \lambda \| \theta \|_1

then this term is like an unscaled Laplace distribution. In practice, this
regularizer encourages many of the model *parameters* to be zero.

In ``theanets``, this sparse parameter regularization is specified using the
``weight_l1`` keyword argument during training::

  net.train(..., weight_l1=1e-4)

The value of the argument is the strength of the regularizer in the loss for the
model. The larger the regularization parameter, the more pressure for
zero-valued weights.

The second type of sparsity regularization puts pressure on the model to develop
hidden *representations* that are mostly zero-valued. In this type of
regularization, the model weights are penalized indirectly, since the hidden
representation (i.e., the values of the hidden layer neurons in the network) are
functions of both the model weights and the input data. In terms of loss
functions, this regularizer adds a term to the loss that penalizes the
:math:`L_1` norm of the hidden layer activations

.. math::
   \mathcal{L}(\cdot) = \dots + \lambda \sum_{i=2}^{N-1} \| f_i(x) \|_1

where :math:`f_i(x)` represents the neuron activations of hidden layer
:math:`i`.

Sparse hidden activations have shown much promise in computational neural
networks. In ``theanets`` this type of regularization is specified using the
``hidden_l1`` keyword argument during training::

  net.train(..., hidden_l1=0.1)

The value of the argument is the strength of the regularizer in the loss for the
model. Large values create more pressure for hidden representations that use
mostly zeros.

Noise
-----

Another way of regularizing a model to prevent overfitting is to inject noise
into the data or the representations during training. While noise could always
be injected into the training batches manually, ``theanets`` provides two types
of noise regularizers: additive Gaussian noise and multiplicative dropout
(binary) noise.

In one method, zero-mean Gaussian noise is added to the input data or hidden
representations. These are specified during training using the ``input_noise``
and ``hidden_noise`` keyword arguments, respectively::

  net.train(..., input_noise=0.1)
  net.train(..., hidden_noise=0.1)

The value of the argument specifies the standard deviation of the noise.

In the other input regularization method, some of the inputs are randomly set to
zero during training (this is sometimes called "dropout" or "multiplicative
masking noise"). This type of noise is specified using the ``input_dropout`` and
``hidden_dropout`` keyword arguments, respectively::

  net.train(..., input_dropout=0.3)
  net.train(..., hidden_dropout=0.3)

The value of the argument specifies the fraction of values in each input or
hidden activation that are randomly set to zero.

Instead of adding additional terms like the other regularizers, the noise
regularizers can be seen as modifying the original loss for a model. For
instance, consider an autoencoder model with two hidden layers::

  net = theanets.Autoencoder([
      100,
      dict(size=50, name='a'),
      dict(size=80, name='b'),
      dict(size=100, name='o')])

The loss for this model, without regularization, can be written as:

.. math::
   \mathcal{L}(X, \theta_a, \theta_b, \theta_o) = \frac{1}{mn} \sum_{i=1}^m \left\|
      \sigma_b(\sigma_a(x_i\theta_a)\theta_b)\theta_o - x_i \right\|_2^2

where we've ignored the bias terms, and :math:`\theta_a`, :math:`\theta_b`, and
:math:`\theta_o` are the parameters for layers a, b, and o, respectively. Also,
:math:`\sigma_a` and :math:`\sigma_b` are the activation functions for their
respective hidden layers.

If we train this model using input and hidden noise::

  net.train(..., input_noise=q, hidden_noise=r)

then the loss becomes:

.. math::
   \mathcal{L}(X, \theta_a, \theta_b, \theta_o) = \frac{1}{mn} \sum_{i=1}^m \left\|
      \left( \sigma_b\left(
      (\sigma_a((x_i+\epsilon_q)\theta_a)+\epsilon_r)\theta_b \right) +
      \epsilon_r \right)\theta_o - x_i \right\|_2^2

where :math:`\epsilon_q` is white Gaussian noise drawn from
:math:`\mathcal{N}(0, qI)` and :math:`\epsilon_r` is white Gaussian noise drawn
separately for each hidden layer from :math:`\mathcal{N}(0, rI)`. The additive
noise pushes the data and the representations off of their respective manifolds,
but the loss is computed with respect to the uncorrupted input. This is thought
to encourage the model to develop representations that push towards the true
manifold of the data.

.. _training-training:

Training
========

.. _training-iteration:

Training as Iteration
---------------------

The :func:`Network.train() <theanets.graph.Network.train>` method is actually
just a thin wrapper over the underlying :func:`Network.itertrain()
<theanets.graph.Network.itertrain>` method, which you can use directly if you
want to do something special during training::

  for train, valid in net.itertrain(train_data, valid_data, **kwargs):
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

The :class:`Network <theanets.graph.Network>` base class can snapshot your model
automatically during training. When you call :func:`Network.train()
<theanets.graph.Network.train>`, you can provide the following keyword
arguments:

- ``save_progress``: This should be a string containing a filename where the
  model should be saved. If you want to save models in separate files during
  training, you can include an empty format string ``{}`` in your filename, and
  it will be formatted with the UTC Unix timestamp at the moment the model is
  saved.

- ``save_every``: This should be a numeric value specifying how often the model
  should be saved during training. If this value is an integer, it specifies the
  number of training iterations between checkpoints; if it is a float, it
  specifies the number of minutes that are allowed to elapse between
  checkpoints.

You can also save and load models manually by calling :func:`Network.save()
<theanets.graph.Network.save>` and :func:`theanets.load()
<theanets.graph.load>`, respectively.
