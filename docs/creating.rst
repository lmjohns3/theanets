.. _creating:

==================
Creating a Network
==================

The first step in using ``theanets`` is creating a neural network model to train
and use. A network model basically consists of three parts:

- an error function that quantifies how well the parameters in the model perform
  for the desired task,
- a set of symbolic input variables that represent data required to compute the
  error, and
- a series of layers that map input data to network outputs.

In ``theanets``, a network model is a subclass of :class:`Network
<theanets.graph.Network>`. Networks encapsulate the first two of these three
parts by implementing the :func:`Network.error()
<theanets.feedforward.Network.error>` method and creating the variables that the
model requires during training. When you choose one of the network models
available in ``theanets`` (or if you choose to make your own), you are choosing
these two parts of the model.

The third part, specifying the layers in the model, is discussed at length
below. First, though, we'll go through a brief presentation of the predefined
network models in ``theanets``. In the brief discussion below, we assume that
the network has some set of parameters :math:`\theta`. In the feedforward pass,
the network computes some function of its inputs :math:`x \in \mathbb{R}^n`
using these parameters; we represent this feedforward function using the
notation :math:`F_\theta(x)`.

.. _creating-predefined-models:

Feedforward Models
==================

There are three major types of neural network models, each defined primarily by
the loss function that the model attempts to optimize. While other types of
models are certainly possible, ``theanets`` only tries to handle the common
cases with built-in model classes. (If you want to define a new type of model,
see :ref:`creating-customizing`.)

Autoencoder
-----------

An :class:`autoencoder <theanets.feedforward.Autoencoder>` takes an array of
arbitrary data vectors :math:`X \in \mathbb{R}^{m \times n}` as input,
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
in practice this doesn't actually happen very often. In addition, a regularizer
:math:`R(X, \theta)` can be added to the overall loss for the model to prevent
this sort of trivial solution.

To create an autoencoder in ``theanets``, just create an instance of the
appropriate network subclass::

  net = theanets.Autoencoder()

Regression
----------

A :class:`regression <theanets.feedforward.Regressor>` model is much like an
autoencoder. Like an autoencoder, a regression model takes as input an array of
arbitrary data :math:`X \in \mathbb{R}^{m \times n}`. However, at training time,
a regression model also requires an array of expected target outputs :math:`Y
\in \mathbb{R}^{m \times o}`. Like an autoencoder, the error between the
network's output and the target is computed using the mean squared error:

.. math::
   \mathcal{L}(X, Y, \theta) = \frac{1}{mn} \sum_{i=1}^m \left\|
      F_\theta(x_i) - y_i \right\|_2^2 + R(X, \theta)

To create a regression model in theanets, invoke the constructor::

  net = theanets.Regressor()

Classification
--------------

A :class:`classification <theanets.feedforward.Classifier>` model takes as input
some piece of data that you want to classify (e.g., the pixels of an image, word
counts from a document, etc.) and outputs a probability distribution over
available labels. At training time, this type of model requires an array of
input data :math:`X \in \mathbb{R}^{m \times n}` and a corresponding set of
integer labels :math:`Y \in \{1,\dots,k\}^m`; the error is then computed as the
cross-entropy between the network output and the true target labels:

.. math::
   \mathcal{L}(X, Y, \theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^k
      \delta_{j,y_i} \log F_\theta(x_i)_j + R(X, \theta)

where :math:`\delta{a,b}` is the Kronecker delta, which is 1 if :math:`a=b` and
0 otherwise.

To create a classifier model in ``theanets``, invoke its constructor::

  net = theanets.Classifier()

.. _creating-recurrent-models:

Recurrent Models
================

The three types of feedforward models described above also exist in recurrent
formulations. In recurrent networks, time is an explicit part of the model. In
``theanets``, if you wish to include recurrent layers in your model, you must
use a model class from the :mod:`theanets.recurrent` module; this is because
recurrent models require input and output data matrices with an additional
dimension to represent time. In general,

- the data shapes required for a recurrent layer are all one
  dimension larger than the corresponding shapes for a feedforward network,

- the extra dimension represents time, and

- the extra dimension is located on:

  - the first (0) axis for versions through 0.6, or
  - the second (1) axis for versions 0.7 and up.

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
\mathbb{R}^{t \times m \times n}` and attempts to recreate the same data at the
output, under a squared-error loss.

Regression
----------

A :class:`recurrent regression <theantes.recurrent.Regressor>` model is also
just like its feedforward counterpart. It requires two inputs at training time:
an array of input data :math:`X \in \mathbb{R}^{t \times m \times n}` and a
corresponding array of output data :math:`Y \in \mathbb{R}^{t \times m \times
o}`. Like the feedforward regression models, the recurrent version attempts to
produce the target outputs under a squared-error loss.

Classification
--------------

A :class:`recurrent classification <theanets.recurrent.Classifier>` model is
like a feedforward classifier in that it takes as input some piece of data that
you want to classify (e.g., the pixels of an image, word counts from a document,
etc.) and outputs a probability distribution over available labels. Computing
the error for this type of model requires an input dataset :math:`X \in
\mathbb{R}^{t \times m \times n}` and a corresponding set of integer labels
:math:`Y \in \mathbb{Z}^{t \times m}`; the error is then computed as the
cross-entropy between the network output and the target labels.

.. _creating-specifying-layers:

Specifying Layers
=================

One of the most critical bits of creating a neural network model is specifying
how the layers of the network are configured. There are very few limits to the
complexity of possible neural network architectures. However, ``theanets`` tries
to make it easy to create networks composed of a cycle-free graph of many common
types of layers.

When you create a network model, the ``layers`` argument is used to specify the
layers for your network. This argument must be a sequence of values, each of
which specifies the configuration of a single layer in the model::

  theanets.Autoencoder([A, B, ..., Z])

Here, the ``A`` through ``Z`` variables represent layer configuration settings.
These variables can be provided using many different types of data. For each of
these configuration variables, ``theanets`` will create a layer with the
corresponding properties, by calling :func:`Network.add_layer()
<theanets.graph.Network.add_layer>`.

Before describing the data types that can be used to configure layers, we will
first describe the configuration values that are being set. Then, we'll go over
the different types of layer configurations, showing how each configuration type
sets the relevant parameters.

Common Parameters
-----------------

Each layer configuration variable provides values for one or more of the
following:

- ``size``: The number of neurons in a layer. This parameter is required for all
  layers.

- ``name``: A string name for the layer. If this isn't provided, the layer will
  be assigned a default name. The default names for the first and last layers in
  a model are "in" and "out" respectively, and the layers in between are
  assigned "hidN" where N is the number of existing layers.

- ``form``: A string specifying the type of layer to use. This defaults to
  "feedforward" but can be the name of any existing :class:`Layer
  <theanets.layers.base.Layer>` subclass.

- ``activation``: A string describing the :ref:`creating-activation` to use for
  the layer. This defaults to "relu".

- ``inputs``: An integer or dictionary describing the sizes of the inputs that
  this layer expects. This is normally optional and defaults to the size of the
  preceding layer in the model. However, providing a dictionary here permits
  arbitrary layer interconnections. See :ref:`creating-graph` for more details.

In addition to these configuration values, each layer can also be provided with
keyword arguments specific to that layer. For example, the distribution of
initial parameter values can be controlled using parameters like ``mean`` and
``sparsity``. See the :class:`Layer <theanets.layers.base.Layer>` class for more
information.

Input Layer
-----------

The first element in the layers configuration sequence is normally a single
integer specifying the size of the expected input data. The ``theanets`` code
creates an :class:`Input <theanets.layers.Input>` layer from this integer value.

The input layer in a model is almost a no-op. It doesn't have any learnable
parameters, and effectively it just passes data along to the first hidden layer.
However, during training, the input layer can inject noise or dropouts into the
data. If you are using an autoencoder model, adding noise at the input creates a
model known as a denoising autoencoder. See
:ref:`training-specifying-regularizers` for more information.

Layer specifications
--------------------

For all subsequent layers (i.e., layers other than the input), there are four
options for each of the layer configuration values.

Layer instances
~~~~~~~~~~~~~~~

If a value in the sequence is a :class:`Layer <theanets.layers.base.Layer>`
instance, this layer is simply added to the network model as-is.

Integers
~~~~~~~~

If a layer value is an integer, that value is interpreted as the ``size`` of a
regular :class:`Feedforward <theanets.layers.feedforward.Feedforward>` layer.
All options for the layer are set to their defaults (e.g., the activation
function defaults to "relu").

For example, to create a network with an input layer containing 4 units, hidden
layers with 5 and 6 units, and an output layer with 2 units, you can just use
integers to specify all of your layers::

  theanets.Classifier((4, 5, 6, 2))

Tuples
~~~~~~

Sometimes you will want to specify more than just the size of a layer. Often a
tuple is a good choice. If a layer configuration value is a tuple, it must
contain an integer and may contain one or more strings.

The integer in the tuple specifies the ``size`` of the layer.

If there is a string in the tuple that names a valid layer type (e.g.,
``'tied'``, ``'rnn'``, etc.), then this type of layer will be created.

If there is a string in the tuple and it does not name a valid layer type, the
string is assumed to name an activation function (e.g., ``'logistic'``,
``'relu+norm:z'``, etc.) and a standard feedforward layer will be created with
that activation.

For example, to create a classification model with a rectified linear activation
in the middle layer and a softmax output layer::

  theanets.Classifier((4, (5, 'relu'), (6, 'softmax')))

Or to create a recurrent model with a vanilla :class:`RNN
<theanets.layers.recurrent.RNN>` middle layer::

  theanets.recurrent.Classifier((4, (5, 'rnn'), (6, 'softmax')))

Note that recurrent models (that is, models containing recurrent layers) are a
bit different from feedforward ones; please see :ref:`creating-recurrent-models`
for more details.

Dictionaries
~~~~~~~~~~~~

If a layer configuration is a dictionary, its keyword arguments are basically
passed directly to :func:`theanets.Layer.build()
<theanets.layers.base.Layer.build>`. The dictionary must contain a ``form`` key,
which specifies the name of the layer type to build, as well as a ``size`` key,
which specifies the number of units in the layer. It can additionally contain
any other keyword arguments that you wish to use when constructing the layer.

For example, you can use a dictionary to specify an non-default activation
function for a layer in your model::

  theanets.Regressor(layers=(4, dict(size=5, activation='tanh'), 2))

You could also create a layer with a sparsely-initialized weight matrix by
providing the ``sparsity`` key::

  theanets.Regressor(layers=(4, dict(size=5, sparsity=0.9), 2))

.. _creating-activation:

Activation Functions
--------------------

An activation function (sometimes also called a transfer function) specifies how
the final output of a layer is computed from the weighted sums of the inputs. By
default, hidden layers in ``theanets`` use a relu activation function. Output
layers in :class:`Regressor <theanets.feedforward.Regressor>` and
:class:`Autoencoder <theanets.feedforward.Autoencoder>` models use linear
activations (i.e., the output is just the weighted sum of the inputs from the
previous layer), and the output layer in :class:`Classifier
<theanets.feedforward.Classifier>` models uses a softmax activation.

To specify a different activation function for a layer, include an activation
key chosen from the table below. As described above, this can be included in
your model specification either using the ``activation`` keyword argument in a
layer dictionary, or by including the key in a tuple with the layer size::

  theanets.Autoencoder([10, (10, 'tanh'), 10])

=========  ============================  ===============================================
Key        Description                   :math:`g(z) =`
=========  ============================  ===============================================
linear     linear                        :math:`z`
sigmoid    logistic sigmoid              :math:`(1 + e^{-z})^{-1}`
logistic   logistic sigmoid              :math:`(1 + e^{-z})^{-1}`
tanh       hyperbolic tangent            :math:`\tanh(z)`
softplus   smooth relu approximation     :math:`\log(1 + \exp(z))`
softmax    categorical distribution      :math:`e^z / \sum e^z`
relu       rectified linear              :math:`\max(0, z)`
trel       truncated rectified linear    :math:`\max(0, \min(1, z))`
trec       thresholded rectified linear  :math:`z \mbox{ if } z > 1 \mbox{ else } 0`
tlin       thresholded linear            :math:`z \mbox{ if } |z| > 1 \mbox{ else } 0`
rect:min   truncation                    :math:`\min(1, z)`
rect:max   rectification                 :math:`\max(0, z)`
norm:mean  mean-normalization            :math:`z - \bar{z}`
norm:max   max-normalization             :math:`z / \max |z|`
norm:std   variance-normalization        :math:`z / \mathbb{E}[(z-\bar{z})^2]`
norm:z     z-score normalization         :math:`(z-\bar{z}) / \mathbb{E}[(z-\bar{z})^2]`
=========  ============================  ===============================================

Activation functions can also be composed by concatenating multiple function
names togather using a ``+``. For example, to create a layer that uses a
batch-normalized hyperbolic tangent activation::

  theanets.Autoencoder([10, (10, 'tanh+norm:z'), 10])

Just like function composition, the order of the components matters! Unlike the
notation for mathematical function composition, the functions will be applied
from left-to-right.

.. _creating-using-weighted-targets:

Using Weighted Targets
======================

By default, the network models available in ``theanets`` treat all inputs as
equal when computing the loss for the model. For example, a regression model
treats an error of 0.1 in component 2 of the output just the same as an error of
0.1 in component 3, and each example of a minibatch is treated with equal
importance when training a classifier.

However, there are times when all inputs to a neural network model are not to be
treated equally. This is especially evident in recurrent models: sometimes, the
inputs to a recurrent network might not contain the same number of time steps,
but because the inputs are presented to the model using a rectangular minibatch
array, all inputs must somehow be made to have the same size. One way to address
this would be to cut off all inputs at the length of the shortest input, but
then the network is not exposed to all input/output pairs during training.

Weighted targets can be used for any model in ``theanets``. For example, an
:class:`autoencoder <theanets.feedforward.Autoencoder>` could use an array of
weights containing zeros and ones to solve a matrix completion task, where the
input array contains some "unknown" values. In such a case, the network is
required to reproduce the known values exactly (so these could be presented to
the model with weight 1), while filling in the unknowns with statistically
reasonable values (which could be presented to the model during training with
weight 0).

As another example, suppose a :class:`classifier
<theanets.feedforward.Classifier>` model is being trained in a binary
classification task where one of the classes---say, class A---is only present
0.1% of the time. In such a case, the network can achieve 99.9% accuracy by
always predicting class B, so during training it might be important to ensure
that errors in predicting A are "amplified" when computing the loss. You could
provide a large weight for training examples in class A to encourage the model
not to miss these examples.

All of these cases are possible to model in ``theanets``; just include
``weighted=True`` when you create your model::

  net = theanets.recurrent.Autoencoder([3, (10, 'rnn'), 3], weighted=True)

When training a weighted model, the training and validation datasets require an
additional component: an array of floating-point values with the same shape as
the expected output of the model. For example, a non-recurrent Classifier model
would require a weight vector with each minibatch, of the same shape as the
labels array, so that the training and validation datasets would each have three
pieces: ``sample``, ``label``, and ``weight``. Each value in the weight array is
used as the weight for the corresponding error when computing the loss.

.. _creating-customizing:

Customizing
===========

The ``theanets`` package tries to strike a balance between defining everything
known in the neural networks literature, and allowing you as a programmer to
create new and exciting stuff with the library. For many off-the-shelf use
cases, the hope is that something in ``theanets`` will work with just a few
lines of code. For more complex cases, you should be able to create an
appropriate subclass and integrate it into your workflow with just a little more
effort.

.. _creating-custom-layers:

Defining Custom Layers
----------------------

Layers are the real workhorse in ``theanets``; custom layers can be created to
do all sorts of fun stuff. To create a custom layer, just create a subclass of
:class:`Layer <theanets.layers.base.Layer>` and give it the functionality you
want.

As a very simple example, let's suppose you wanted to create a normal
feedforward layer but did not want to include a bias term::

  import theanets
  import theano.tensor as TT

  class NoBias(theanets.Layer):
      def transform(self, inputs):
          return TT.dot(inputs, self.find('w'))

      def setup(self):
          self.add_weights('w')

Once you've set up your new layer class, it will automatically be registered and
available in :func:`theanets.Layer.build <theanets.layers.base.Layer.build>`
using the name of your class::

  layer = theanets.Layer.build('nobias', inputs=3, size=4)

or, while creating a model::

  net = theanets.Autoencoder(
      layers=(4, (3, 'nobias', 'linear'), (4, 'tied', 'linear')),
  )

This example shows how fast it is to create a PCA-like model that will learn the
subspace of your dataset that spans the most variance---the same subspace
spanned by the principal components.

.. _creating-custom-regularizers:

Defining Custom Regularizers
----------------------------

To create a custom regularizer in ``theanets``, you need to create a custom
subclass of the :class:`Regularizer <theanets.regularizers.Regularizer>` class,
and then provide this regularizer when you run your model.

To illustrate, let's continue with the example above. Suppose you created a
linear autoencoder model that had a larger hidden layer than your dataset::

  net = theanets.Autoencoder([4, (8, 'linear'), (4, 'tied')])

Then, at least in theory, you risk learning an uninteresting "identity" model
such that some hidden units are never used, and the ones that are have weights
equal to the identity matrix. To prevent this from happening, you can impose a
sparsity penalty when you train your model::

  net.train(..., hidden_l1=0.001)

But then you might run into a situation where the sparsity penalty drives some
of the hidden units in the model to zero, to "save" loss during training.
Zero-valued features are probably not so interesting, so we can introduce
another penalty to prevent feature weights from going to zero::

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

.. _creating-custom-errors:

Defining Custom Error Functions
-------------------------------

It's pretty straightforward to create models in ``theanets`` that use different
error functions from the predefined :class:`Classifier
<theanets.feedforward.Classifier>` and :class:`Autoencoder
<theanets.feedforward.Autoencoder>` and :class:`Regressor
<theanets.feedforward.Regressor>` models. (The classifier uses categorical
cross-entropy (XE) as its default loss, and the other two both use mean squared
error, MSE.)

To define a model with a new loss, just create a new :class:`Loss
<theanets.losses.Loss>` subclass and specify its name when you create your
model. For example, to create a regression model that uses a step function
averaged over all of the model inputs::

  class Step(theanets.Loss):
      def __call__(self, outputs):
          return (self.diff(outputs) > 0).mean()

  net = theanets.Regressor(loss='step')

Your loss function implementation must return a theano expression that reflects
the loss for your model. If you wish to make your loss work with weighted
outputs, you will also need to include a case for having weights::

  class Step(theanets.Loss):
      def __call__(self, outputs):
          step = self.diff(outputs) > 0
          if self.weight:
              return (self.weight * step).sum() / self.weight.sum()
          else:
              return step.mean()

.. _creating-graph:

Creating a Graph
================

While many types of neural networks are constructed using a single linear
"stack" of layers, this does not always need to be the case. Indeed, many of the
more exotic model types that perform well in specialized settings make use of
connections between multiple inputs and outputs.

In ``theanets`` it is easiest to create network architectures that use a single
chain of layers. However, it is also possible to create network graphs that have
arbitrary, acyclic connections among layers. Creating a nonlinear network graph
requires using the ``inputs`` keyword argument when creating a layer.

The ``inputs`` keyword argument for creating a layer should be a dictionary that
maps from the name of a network output to the size of that output. If ``inputs``
is not specified for a layer, ``theanets`` creates a default dictionary that
just uses the output from the previous layer.

Perhaps the simplest example of a non-default ``inputs`` dictionary is to create
a classifier model that uses outputs from all hidden layers to inform the final
output of the layer. Such a "multi-scale" model can be created as follows::

  theanets.Classifier((
      784,
      dict(size=100, name='a'),
      dict(size=100, name='b'),
      dict(size=100, name='c'),
      dict(size=10, inputs={'a:out': 100, 'b:out': 100, 'c:out': 100}),
  ))

Here, each of the hidden layers is assigned an explicit name, so that they will
be easy to reference by the last layer. The output layer, a vanilla feedforward
layer, combines together the outputs from layers ``a``, ``b``, and ``c``.
