.. _regularizers:

============
Regularizers
============

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
model:

.. code:: python

  net = theanets.Classifier(layers=[784, 1000, 784])
  net.train(..., hidden_l1=0.1)

Here we've specified that our model has a single, overcomplete hidden layer, and
then when we train it, we specify that the activity of the hidden units in the
network will be penalized with a 0.1 coefficient. The rest of this section
details the built-in regularizers that are available in ``theanets``.

Decay
=====

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
argument during training:

.. code:: python

  net.train(..., weight_l2=1e-4)

The value of the argument is the strength of the regularizer in the loss for the
model. Larger values create more pressure for small model weights.

Sparsity
========

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
``weight_l1`` keyword argument during training:

.. code:: python

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
``hidden_l1`` keyword argument during training:

.. code:: python

  net.train(..., hidden_l1=0.1)

The value of the argument is the strength of the regularizer in the loss for the
model. Large values create more pressure for hidden representations that use
mostly zeros.

Noise
=====

Another way of regularizing a model to prevent overfitting is to inject noise
into the data or the representations during training. While noise could always
be injected into the training batches manually, ``theanets`` provides two types
of noise regularizers: additive Gaussian noise and multiplicative dropout
(binary) noise.

In one method, zero-mean Gaussian noise is added to the input data or hidden
representations. These are specified during training using the ``input_noise``
and ``hidden_noise`` keyword arguments, respectively:

.. code:: python

  net.train(..., input_noise=0.1)
  net.train(..., hidden_noise=0.1)

The value of the argument specifies the standard deviation of the noise.

In the other input regularization method, some of the inputs are randomly set to
zero during training (this is sometimes called "dropout" or "multiplicative
masking noise"). This type of noise is specified using the ``input_dropout`` and
``hidden_dropout`` keyword arguments, respectively:

.. code:: python

  net.train(..., input_dropout=0.3)
  net.train(..., hidden_dropout=0.3)

The value of the argument specifies the fraction of values in each input or
hidden activation that are randomly set to zero.

Instead of adding additional terms like the other regularizers, the noise
regularizers can be seen as modifying the original loss for a model. For
instance, consider an autoencoder model with two hidden layers:

.. code:: python

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

If we train this model using input and hidden noise:

.. code:: python

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

Predefined Regularizers
=======================

.. automodule:: theanets.regularizers
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Regularizer
   HiddenL1
   WeightL1
   WeightL2
   Contractive
   RecurrentNorm
   RecurrentState
   BernoulliDropout
   GaussianNoise

.. _regularizers-custom:

Custom Regularizers
===================

To create a custom regularizer in ``theanets``, you need to create a custom
subclass of the :class:`theanets.Regularizer
<theanets.regularizers.Regularizer>` class, and then provide this regularizer
when you run your model.

To illustrate, let's suppose you created a linear autoencoder model that had a
larger hidden layer than your dataset:

.. code:: python

  net = theanets.Autoencoder([4, (8, 'linear'), (4, 'tied')])

Then, at least in theory, you risk learning an uninteresting "identity" model
such that some hidden units are never used, and the ones that are have weights
equal to the identity matrix. To prevent this from happening, you can impose a
sparsity penalty when you train your model:

.. code:: python

  net.train(..., hidden_l1=0.001)

But then you might run into a situation where the sparsity penalty drives some
of the hidden units in the model to zero, to "save" loss during training.
Zero-valued features are probably not so interesting, so we can introduce
another penalty to prevent feature weights from going to zero:

.. code:: python

  class WeightInverse(theanets.Regularizer):
      def loss(self, layers, outputs):
          return sum((1 / (p * p).sum(axis=0)).sum()
                     for l in layers for p in l.params
                     if p.ndim == 2)

  net = theanets.Autoencoder([4, (8, 'linear'), (4, 'tied')])
  net.train(..., hidden_l1=0.001, weightinverse=0.001)

This code adds a new regularizer that penalizes the inverse of the squared
length of each of the weights in the model's layers. Here we detect weights by
only including parameters with 2 dimensions.
