==========
User Guide
==========

.. _guide-existing:

Using Existing Models
=====================

Several broad classes of models are pre-defined in ``theanets``:

- :ref:`Classifier <models-classification>`: A model that maps its input onto a
  (usually small) set of output nodes that represent the probability of a label
  given the input.
- :ref:`Autoencoder <models-autoencoders>`: A model that attempts to reproduce
  its input as its output.
- :ref:`Regressor <models-regression>`: Like the classifier, but instead of
  attempting to produce a `one-hot`_ output label, a regressor attempts to
  produce some continuous-valued target vector for each input.

.. _one-hot: http://en.wikipedia.org/wiki/One-hot

It's also pretty simple to create custom models using ``theanets``; see
:ref:`hacking-extending` for more information.

.. _guide-model-hyperparameters:

Model Hyperparameters
=====================

By default, layers in ``theano`` are constructed using straightforward
:class:`feedforward <theanets.layers.Feedforward>` layers; these layers compute
a weighted (affine) transformation of their input, and then perform a point-wise
(i.e., independent on each computation unit) nonlinear transform.

Activation functions
--------------------

- linear
- logistic sigmoid
- hyperbolic tangent
- rectified linear
- softplus
- softmax

Regularizers
------------

If you want to set up a more sophisticated model like a classifier with sparse
hidden representations, you can add regularization hyperparameters when you
create your experiment::

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(784, 1000, 784),
      hidden_l1=0.1)

Here we've specified that our model has a single, overcomplete hidden layer, and
the activity of the hidden units in the network will be penalized with a 0.1
coefficient.

.. _guide-training-hyperparameters:

Training Hyperparameters
========================

Training as iteration
`````````````````````

The :func:`Experiment.train() <theanets.main.Experiment.train>` method is
actually just a thin wrapper over the underlying :func:`Experiment.itertrain()
<theanets.main.Experiment.itertrain>` method, which you can use directly if you
want to do something special during training::

  for monitors in exp.itertrain(train, valid, **kwargs):
      print(monitors['loss'])

Trainers yield a dictionary after each training iteration. The keys and values
in each dictionary give the costs and monitors that are computed during
training, which will vary depending on the model and the training algorithm.
However, there will always be a ``'loss'`` key that gives the value of the loss
function that is being optimized. For classifier models, the dictionary will
also have an ``'acc'`` key, which gives the percent accuracy of the classifier
model.

.. note::
   The :class:`HF <theanets.trainer.HF>` trainer and the :class:`Sample
   <theanets.trainer.Sample>` trainer always return loss values equal to -1.

.. _guide-autoencoders:

Autoencoders
------------

The ``theanets`` package also provides an :class:`Autoencoder
<theanets.feedforward.Autoencoder>` class to construct models that can learn
features from data without labels. An autoencoder for MNIST digits, for example,
takes as input an unlabeled MNIST digit image and then attempts to produce this
same digit image as output. The hidden layers in such a model are then called
the "features" of the data that the model learns.

An autoencoder must always have the same number of inputs as outputs. The output
layer typically has a linear activation, which treats the data as a weighted sum
of some fixed set of *basis vectors* that spans the space of the data being
modeled. For an MNIST autoencoder task, your model must have 784 inputs and 784
outputs.

There can be any number of layers between the input and output, and they can be
of practically any form, but there are a few notable classes of autoencoders:

- *Undercomplete autoencoders* (also called *bottleneck autoencoders*) have a
  hidden layer that is smaller than the input layer. A small hidden layer is
  referred to as a bottleneck because the model must find some way to compress
  the input data into a smaller-dimensional space without losing too much
  information.

- *Overcomplete autoencoders* have hidden layers that are all larger than the
  input layer. These models are capable of learning a trivial identity transform
  from the inputs to the hidden layer(s) and on to the outputs, so they are
  often *regularized* in various ways to learn robust features.

  For example, a :ref:`sparse autoencoder <models-sparse-autoencoder>` is
  penalized for using large values in the hidden-unit activations, and a
  :ref:`denoising autoencoder <models-denoising-autoencoder>` adds noise to the
  inputs and forces the model to reconstruct the noise-free inputs.

- As with classifiers, *deep autoencoders* are any autoencoder model with more
  than a small number of hidden layers. Deep models have been quite popular
  recently, as they perform quite well on a variety of difficult machine
  learning tasks.

.. note::
   Command-line arguments do not work when running ``theanets`` code in
   IPython; within IPython, all parameters must be specified as keyword
   arguments.

Finally, a subset of autoencoders with an odd-length, palindromic number of
layers can be defined as having **tied weights** whenever the parameters
from the decoder are the transpose of the parameters from the encoder.
Tied-weights autoencoders form an interesting subset of autoencoder models.

Let's look at a few example models that fall into the autoencoder class.

Single-layer autoencoders
-------------------------

Although the class of autoencoder models is quite large (any :math:`k`
partite graph like the one described above, having the same number of input
and output nodes would count). However, a very interesting class of these
models has just one hidden layer, and uses a linear activation on its output
nodes:

.. math::
   F_\theta(x) = \sigma(x W_e + b_e) W_d + b_d

Here, :math:`\sigma` is the activation of the nodes in the hidden layer, and
:math:`W_e`, :math:`W_d`, :math:`b_e`, and :math:`b_d` are the weights and
bias of the "encoding" and "decoding" layers of the network. The trainable
parameters are :math:`\theta = (W_e, W_d, b_e, b_d)`.

To train the weights and biases in the network, an autoencoder typically
optimizes a squared-error reconstruction loss:

.. math::
   J(X, \theta) = \frac{1}{M} \sum_{i=1}^M \left\| \sigma(x_i W_e + b_e) W_d + b_d - x_i \right\|_2^2 + \lambda R(X, \theta)

This optimization process could result in a trivial model, depending on the
setup of the network. In particular, if the number of hidden features
:math:`n_z` is not less than the number of input variables :math:`n_0`, then
with linear hidden activations :math:`\sigma(z) = z`, identity weights
:math:`W_e = W_d = I`, and zero bias :math:`b_e = b_d = 0`, an autoencoder
as defined above implements the identity transform:

.. math::
   F_\theta(x) = x

Even if the hidden unit activations are nonlinear, the network is capable of
learning an identity transform as long as :math:`n_z \ge n_0`. But things
get much more interesting when an autoencoder network is forced to reproduce
the input under some constraint. These constraints can be implemented either
through the structure of the network, or by adding a regularizer. Both of
these approaches will be discussed below.

PCA
```

One way to prevent a model from learning trivial latent representations is
to force the latent space to be smaller than the space where the data live.
One of the most popular techniques for doing this is Principal Component
Analysis (PCA) [Hot33]_. The principal components (PCs) of a dataset are the
set of orthogonal directions :math:`U` (i.e., a rotation) that capture the
maximal variance in a dataset. Each PC :math:`u_i` is scaled by the amount
of variance :math:`s_i` in the corresponding direction of the data, so the
first PC captures the most variance, the second PC the second-most variance,
and so forth.

Let's assume we have computed a PCA transform :math:`W = UD_s` for a dataset
:math:`X` (here, :math:`D_s` is a diagonal matrix with the :math:`s_i` along
the diagonal). Then we can "encode" the dataset by projecting it into the PC
space using matrix multiplication to rotate and then scale the data:

.. math:: Z = XUD_s

If we wish to "decode" this representation of the data, we can project it
back into the data space by doing another matrix multiplication to un-scale
the data and rotate it back:

.. math::
   \hat{X} = ZD_{1/S}U^\top = X U D_s D_{1/S} U^\top = X U U^\top

If we have the same number of principal components as variables in our
dataset, then :math:`UU^\top = I` and :math:`\hat{X} = X`. However, if we
restrict our PC representation to a smaller number of dimensions than we
have in our data, we are performing *dimensionality reduction* in a way that
is guaranteed to preserve the most variance in the data. In other words, our
transform :math:`UD_s` minimizes the squared-error loss:

.. math::
   J(X) = \frac{1}{M} \sum_{i=1}^M \left\| \hat{x}_i - x_i \right\|_2^2

.. math::
   J(X) = \frac{1}{M} \sum_{i=1}^M \left\| x_i U U^\top - x_i \right\|_2^2

Given this way of looking at PCA, we can see that it is really a sort of
linear autoencoder with tied weights! To be more precise, optimizing the
loss formulation immediately above is guaranteed to recover the same
*subspace* as the PCA transform, even though the individual features are not
necessarily guaranteed to be the same.

To implement such a model in ``theanets``, we only need to provide the
following hyperparameters::

  pca = theanets.Experiment(
      theanets.Autoencoder,
      tied_weights=True,
      hidden_activation='linear',
      layers=(n_0, n_z, n_0),
  )

This type of model has the additional advantage that it is relatively easy
to train, because the entire model is linear!

In actuality, if your dataset is not too large, it's even easier to use a
closed-form solution to compute the PCA transform; however, looking at PCA
in this way, using a neural network framework, will serve as a good mental
bridge to the sorts of models that will be introduced later on.

ICA
```

For PCA, we had to use an *undercomplete* hidden representation to prevent
the model from learning a trivial identity transform. This is problematic
for a couple of reasons, but from a modeling perspective one of the worst is
that the features computed by PCA are often "tangled together" to represent
each of the points in our dataset. That is, a single PCA feature is often
difficult to interpret by itself; instead, the entire set of PCs is required
to yield a reasonable representation of a data point.

For example, if PCA is performed on a set of image data, the PCs are
typically close to a Fourier basis for the space of images being processed;
this representation does in fact capture the most variance in the data, but
any individual PC only captures one of the spatial frequencies in an
image---a relatively large part of the entire set of PCs must be used to
reconstruct an image with good fidelity.

If instead we wanted to learn an *overcomplete* feature set (i.e., with
:math:`n_z > n_0`), or if we wanted to learn some features of our data that
were not dependent on the others, we could encourage the model to learn a
non-trivial representation of the data by adding a regularizer that
specifies how the features should behave.

One good intuition for introducing a regularizer at this point is to assume
that latent features should be used independently. We can translate that
into mathematics by requiring that the model reproduce the input data using
"as little" feature representation as possible and add an :math:`L_1`
penalty to the hidden representation:

.. math::
   J(X, W) = \left\| WW^\top x - x \right\|_2^2 + \lambda \left\| W^\top x \right\|_1

This model, called RICA [Le11]_ ("ICA with a reconstruction cost"), is
actually equivalent to an existing statistical model called Independent
Component Analysis [Jut91]_ [Hyv97]_, which can be trained by maximizing the
non-gaussian-ness (e.g., the kurtosis) of the features. Here, we force the
model to use a sparse representation while still using linear encoding and
decoding with tied weights.

In ``theanets``, we can create such a model by including a sparsity penalty
on the hidden layer::

  rica = theanets.Experiment(
      theanets.Autoencoder,
      tied_weights=True,
      hidden_activation='linear',
      hidden_l1=1,
      layers=(n_0, n_z, n_0),
  )

This model does not have a simple closed-form solution, so an iterative
optimization procedure is just what we need to learn good parameters for the
model.

Sparse autoencoders
-------------------

RICA models (and ICA generally) are a subset of a more general class of
autoencoder called a *sparse autoencoder* [Glo11]_. Sparse autoencoders
generalize the RICA formulation by adding:

- different encoding and decoding weights,
- bias terms, and
- a nonlinearity at the hidden layer.

Like RICA, however, sparse autoencoders assign a regularization penalty to
the hidden activation of the model:

.. math::
   J(X, \theta) = \frac{1}{M} \sum_{i=1}^M \left\| \sigma(x_i W_e + b_e) W_d +
   b_d - x_i \right\|_2^2 + \lambda\left\| \sigma(x_i W_e + b_e) \right\|_1

The sparsity penalty forces the encoder and decoder of the autoencoder model
to cooperate together to represent the input using as little of the latent
space as possible.

To create a sparse autoencoder in ``theanets``, just use the RICA
formulation but omit the tied weights and linear activation::

  sparse = theanets.Experiment(
      theanets.Autoencoder,
      hidden_l1=1,
      layers=(n_0, n_z, n_0),
  )

Sparse autoencoders can also be created with more than one hidden layer.

Denoising autoencoders
----------------------

.. _guide-extending:

Creating New Models
===================

.. _guide-extending-regularizers:

Defining Custom Regularizers
----------------------------

.. _guide-extending-costs:

Defining Custom Cost Functions
------------------------------

It's pretty straightforward to create models in ``theanets`` that use cost
functions that are different from the predefined :class:`Classifier
<theanets.feedforward.Classifier>` (which uses binary cross-entropy) and
:class:`Regressor <theanets.feedforward.Regressor>` (which uses mean squared
error). To define by a model with a new cost function, just create a new
subclass and override the ``cost`` property on your subclass. For example, to
create a regression model that uses mean absolute error::

  class MaeRegressor(theanets.Regressor):
      @property
      def cost(self):
          err = self.outputs[-1] - self.targets
          return TT.mean(abs(err).sum(axis=1))

Your cost function must return a theano expression that reflects the cost for
your model.

.. _guide-data:

Providing Data
==============

.. _guide-data-callables:

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

.. _guide-contributing:

More Information
================

This concludes the user guide! You can read more information about ``theanets``
in the :doc:`quickstart` and :doc:`reference` sections of the documentation.

The source code for ``theanets`` lives at http://github.com/lmjohns3/theanets.
Please fork, explore, and send pull requests!

Finally, there is also a mailing list for project discussion and announcements.
Subscribe online at https://groups.google.com/forum/#!forum/theanets.
