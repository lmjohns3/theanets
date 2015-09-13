.. _models:

======
Models
======

There are three major types of neural network models, each defined primarily by
the :ref:`loss function <losses>` that the model attempts to optimize. While
other types of models are certainly possible, ``theanets`` only tries to handle
the common cases with built-in model classes. If you want to define a new type
of model, see :ref:`models-custom`.

To describe the predefined models, we assume that a neural network has some set
of parameters :math:`\theta`. In the feedforward pass, the network computes some
function of an input vector :math:`x \in \mathbb{R}^n` using these parameters;
we represent this feedforward function using the notation :math:`y =
F_\theta(x)`.

Autoencoder
===========

An :class:`autoencoder <theanets.feedforward.Autoencoder>` takes an array of
:math:`m` arbitrary data vectors :math:`X \in \mathbb{R}^{m \times n}` as input,
transforms it in some way, and then attempts to recreate the original input as
the output of the network.

To evaluate the loss for an autoencoder, only the input data is required. The
default autoencoder model computes the loss using the mean squared error between
the network's output and the input:

.. math::
   \mathcal{L}(X, \theta) = \frac{1}{mn} \sum_{i=1}^m \left\|
      F_\theta(x_i) - x_i \right\|_2^2 + R(X, \theta)

Autoencoders simply try to adjust their model parameters :math:`\theta` to
minimize this squared error between the true inputs and the values that the
network produces.

In theory this could be trivial---if, for example, :math:`F_\theta(x) = x`---but
in practice this doesn't actually happen very often. In addition, a
:ref:`regularizer <guide-training-specifying-regularizers>` :math:`R(X, \theta)`
can be added to the overall loss for the model to prevent this sort of trivial
solution.

To create an autoencoder in ``theanets``, just create an instance of the
appropriate network subclass:

.. code:: python

  net = theanets.Autoencoder()

Of course you'll also need to specify which types of layers you'd like in your
model; this is discussed in :ref:`guide-creating-specifying-layers`.

Regression
==========

A :class:`regression <theanets.feedforward.Regressor>` model is much like an
autoencoder. Like an autoencoder, a regression model takes as input an array of
arbitrary data :math:`X \in \mathbb{R}^{m \times n}`. However, at training time,
a regression model also requires an array of expected target outputs :math:`Y
\in \mathbb{R}^{m \times o}`. Like an autoencoder, the error between the
network's output and the target is computed using the mean squared error:

.. math::
   \mathcal{L}(X, Y, \theta) = \frac{1}{mn} \sum_{i=1}^m \left\|
      F_\theta(x_i) - y_i \right\|_2^2 + R(X, \theta)

The difference here is that instead of trying to produce the input, the
regression model is trying to match the target output.

To create a regression model in theanets, just invoke the constructor:

.. code:: python

  net = theanets.Regressor()

Again, you'll need to specify which types of layers you'd like in your model;
this is discussed in :ref:`guide-creating-specifying-layers`.

Classification
==============

A :class:`classification <theanets.feedforward.Classifier>` model takes as input
some piece of data that you want to classify (e.g., the pixels of an image, word
counts from a document, etc.) and outputs a probability distribution over
available labels.

At training time, this type of model requires an array of input data :math:`X
\in \mathbb{R}^{m \times n}` and a corresponding set of integer labels :math:`Y
\in \{1,\dots,k\}^m`. The error is then computed as the cross-entropy between
the network output and the true target labels:

.. math::
   \mathcal{L}(X, Y, \theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^k
      \delta_{j,y_i} \log F_\theta(x_i)_j + R(X, \theta)

where :math:`\delta{a,b}` is the Kronecker delta, which is 1 if :math:`a=b` and
0 otherwise.

To create a classifier model in ``theanets``, invoke its constructor:

.. code:: python

  net = theanets.Classifier()

As with the other models, you'll need to specify which types of layers you'd
like in your model; this is discussed in
:ref:`guide-creating-specifying-layers`.

Recurrent Models
================

The three predefined models described above also exist in recurrent
formulations. In recurrent networks, time is an explicit part of the model. In
``theanets``, if you wish to include recurrent layers in your model, you must
use a model class from the :mod:`theanets.recurrent` module; this is because
recurrent models require input and output data matrices with an additional
dimension to represent time. In general,

- the data shapes required for a recurrent layer are all one
  dimension larger than the corresponding shapes for a feedforward network,

- the extra dimension represents time, and

- the extra dimension is located on:

  - the first (0) axis in ``theanets`` versions through 0.6, or
  - the second (1) axis in ``theanets`` versions 0.7 and up.

.. warning::

   Starting with release 0.7.0 of ``theanets``, recurrent models have changed
   the expected axis ordering for data arrays! The axis ordering before version
   0.7.0 was ``(time, batch, variables)``, and the axis ordering starting in the
   0.7.0 release is ``(batch, time, variables)``.

   The new ordering is more consistent with other models in ``theanets``.
   Starting in the 0.7 release, the first axis (index 0) of data arrays for all
   model types represents the examples in a batch, and the last axis (index -1)
   represents the input variables. For recurrent models, the axis in the middle
   of a batch (index 1) represents time.

.. note::

   In recurrent models, the batch size is currently required to be greater than
   one. If you wish to run a recurrent model on a single sample, just create a
   batch with two copies of the same sample.

Autoencoding
------------

A :class:`recurrent autoencoder <theanets.recurrent.Autoencoder>`, just like its
feedforward counterpart, takes as input a single array of data :math:`X \in
\mathbb{R}^{m \times t \times n}` and attempts to recreate the same data at the
output, under a squared-error loss.

To create a model of this type, just invoke its constructor:

.. code:: python

   net = theanets.recurrent.Autoencoder()

Regression
----------

A :class:`recurrent regression <theantes.recurrent.Regressor>` model is also
just like its feedforward counterpart. It requires two inputs at training time:
an array of input data :math:`X \in \mathbb{R}^{m \times t \times n}` and a
corresponding array of output data :math:`Y \in \mathbb{R}^{m \times t \times
o}`. Like the feedforward regression models, the recurrent version attempts to
produce the target outputs under a squared-error loss.

To create a model of this type, just invoke its constructor:

.. code:: python

   net = theanets.recurrent.Regressor()

Classification
--------------

A :class:`recurrent classification <theanets.recurrent.Classifier>` model is
like a feedforward classifier in that it takes as input some piece of data that
you want to classify (e.g., the pixels of an image, word counts from a document,
etc.) and outputs a probability distribution over available labels. Computing
the error for this type of model requires an input dataset :math:`X \in
\mathbb{R}^{m \times t \times n}` and a corresponding set of integer labels
:math:`Y \in \mathbb{Z}^{t \times m}`; the error is then computed as the
cross-entropy between the network output and the target labels.

To create a model of this type, just invoke its constructor:

.. code:: python

   net = theanets.recurrent.Classifier()

.. _models-custom:

Custom Models
=============

To create a custom model, just define a new subclass of :class:`theanets.Network
<theanets.graph.Network>`.

For instance, the :class:`feedforward autoencoder
<theanets.feedforward.Autoencoder>` model is defined basically like this:

.. code:: python

  class Autoencoder(theanets.Network):
      def __init__(self, layers=(), loss='mse', weighted=False):
          super(Autoencoder, self).__init__(
              layers=layers, loss=loss, weighted=weighted)

Essentially this model just defines a default loss on top of the functionality
in :class:`theanets.Network <theanets.graph.Network>` for creating and managing
layers and loss functions, training the model, making predictions, and so on.

By defining a custom model class, you can also implement whatever helper
functionality you think will be useful for your task. With the programming power
of Python, the sky's the limit!
