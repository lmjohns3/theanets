==================
Creating a Network
==================

The first step in using ``theanets`` is creating a model to train and use.
This basically involves two parts:

- one of the three broad classes of network models, and
- a series of layers that map inputs to outputs.

In ``theanets``, a network model is a subclass of :class:`Network
<theanets.feedforward.Network>`. Its primary defining characteristics are the
implementations of the :func:`Network.error()
<theanets.feedforward.Network.error>` and :func:`Network.setup_vars()
<theanets.feedforward.Network.setup_vars>` methods.

The ``error`` method defines the error function for the model, as a function of
the output of the network, and any internal symbolic variables that the model
defines. The error is an important (and sometimes the only) component of the
loss that model trainers attempt to minimize during the learning process.

The ``setup_vars`` method defines the variables that the network requires for
computing an error value. All variables that are required to compute the error
must be defined in this method.

In the brief discussion below, we assume that the network has some set of
parameters :math:`\theta`. In the feedforward pass, the network computes some
function of its inputs :math:`x \in \mathbb{R}^n` using these parameters; we
represent this feedforward function using the notation :math:`F_\theta(x)`.

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
arbitrary data :math:`x` as input, transforms it in some way, and then attempts
to recreate the original input as the output of the network.

To evaluate the loss for an autoencoder, only the input data is required. The
default autoencoder model computes the loss using the mean squared error between
the network's output and the input:

.. math::
   \mathcal{L}(X, \theta) = \frac{1}{m}\frac{1}{n} \sum_{i=1}^m \left\| F_\theta(x_i) - x_i \right\|_2^2 + R(X, \theta)

To create an autoencoder in theanets, you can create a network class directly::

  net = theanets.Autoencoder()

or you can use an :class:`Experiment <theanets.main.Experiment>`::

  exp = theanets.Experiment(theanets.Autoencoder)
  net = exp.network

An autoencoder requires the following inputs at training time:

- ``x``: A two-dimensional array of input data. Each row of ``x`` is expected to
  be one data item. Each column of ``x`` holds the measurements of a particular
  input variable across all data items.

Regression
----------

A :class:`regression <theanets.feedforward.Regressor>` model is much like an
autoencoder, except that at training time, the expected output :math:`Y \in
\mathbb{R}^{m \times o}` must be provided to the model. Like an autoencoder, a
regression model takes as input an array of arbitrary data :math:`X \in
\mathbb{R}^{m \times n}`, and the difference between the network's output and
the target is computed using the mean squared error:

.. math::
   \mathcal{L}(X, Y, \theta) = \frac{1}{m}\frac{1}{o} \sum_{i=1}^m \left\| F_\theta(x_i) - y_i \right\|_2^2 + R(X, \theta)

To create a regression model in theanets, you can create a network class
directly::

  net = theanets.Regressor()

or you can use an :class:`Experiment <theanets.main.Experiment>`::

  exp = theanets.Experiment(theanets.Regressor)
  net = exp.network

A regression model requires the following inputs at training time:

- ``x``: A two-dimensional array of input data. Each row of ``x`` is expected to
  be one data item. Each column of ``x`` holds the measurements of a particular
  input variable across all data items.
- ``targets``: A two-dimensional array of target output data. Each row of
  ``targets`` is expected to be the target values for a single data item. Each
  column of ``targets`` holds the measurements of a particular output variable
  across all data items.

The number of rows in ``x`` must be equal to the number of rows of ``targets``,
but the number of columns in these two arrays may be whatever is required for
the inputs and outputs of the problem.

Classification
--------------

A :class:`classification <theanets.feedforward.Classifier>` model takes as input
some piece of data that you want to classify (e.g., the pixels of an image, word
counts from a document, etc.) and outputs a probability distribution over
available labels. The error for this type of model takes an input dataset
:math:`X \in \mathbb{R}^{m \times n}` and a corresponding set of integer labels
:math:`Y \in \mathbb{Z}^m`; the error is then computed as the cross-entropy
between the network output and the target labels:

.. math::
   \mathcal{L}(X, Y, \theta) = \frac{1}{m} \sum_{i=1}^m - \log F_\theta(x_i)_{y_i} + R(x, \theta)

To create a classifier model in ``theanets``, you can create a network class
directly::

  net = theanets.Classifier()

or you can use an :class:`Experiment <theanets.main.Experiment>`::

  exp = theanets.Experiment(theanets.Classifier)
  net = exp.network

A classifier model requires the following inputs at training time:

- ``x``: A two-dimensional array of input data. Each row of ``x`` is expected to
  be one data item. Each column of ``x`` holds the measurements of a particular
  input variable across all data items.
- ``labels``: A one-dimensional array of target labels. Each element of
  ``labels`` is expected to be the class index for a single data item.

The number of rows in ``x`` must match the number of elements in the ``labels``
vector. Additionally, the values in ``labels`` are expected to range from 0 to
the number of classes in the data being modeled. For example, for the MNIST
digits dataset, which represents digits 0 through 9, the labels array contains
integer class labels 0 through 9.

.. _creating-recurrent-models:

Recurrent Models
================

The three types of feedforward models described above also exist in recurrent
formulations, but in recurrent networks, time is an explicit part of the model.
In ``theanets``, if you wish to include recurrent layers in your model, you must
use a model class from the :mod:`theanets.recurrent` module; this is because
recurrent models require data matrices with an additional dimension to represent
time. In general,

- the data shapes required for a recurrent layer are all one
  dimension larger than the corresponding shapes for a feedforward network, and
- the extra dimension is always the 0 axis, and
- the extra dimension represents time.

In addition to the three vanilla model types described above, recurrent networks
also allow for the possibility of *predicting future outputs*. This task is
handled by prediction networks.

Autoencoder
-----------

A :class:`recurrent autoencoder <theanets.recurrent.Autoencoder>`, just like its
feedforward counterpart, takes as input a single array of data :math:`X \in
\mathbb{R}^{t \times m \times n}` and attempts to recreate the same data at the
output, under a squared-error loss.

A recurrent autoencoder thus requires the following inputs:

- ``x``: A three-dimensional array of input data. Each element of axis 0 of
  ``x`` is expected to be one sample in time. Each element of axis 1 of ``x``
  holds a single data sample. Each element of axis 2 of ``x`` represents the
  measurements of a particular input variable across all times and all data
  items.

Prediction
----------

An interesting subclass of autoencoders is models that attempt to predict future
states based on past data. :class:`Prediction <theanets.recurrent.Predictor>`
models are like autoencoders in that they require only a data array as input,
and they train under a squared-error loss. Unlike a recurrent autoencoder,
however, a prediction model is explicitly required to produce a future output,
rather than the output from the same time step.

A recurrent prediction model takes the following inputs:

- ``x``: A three-dimensional array of input data. Each element of axis 0 of
  ``x`` is expected to be one sample in time. Each element of axis 1 of ``x``
  holds a single data sample. Each element of axis 2 of ``x`` represents the
  measurements of a particular input variable across all times and all data
  items.

Regression
----------

A recurrent regression model is also just like its feedforward counterpart. It
requires two inputs at training time: an array of input data :math:`X \in
\mathbb{R}^{t \times m \times n}` and a corresponding array of output data
:math:`Y \in \mathbb{R}^{t \times m \times o}`. Like the feedforward regression
models, the recurrent version attempts to produce the target outputs under a
squared-error loss.

A recurrent regression model takes the following inputs:

- ``x``: A three-dimensional array of input data. Each element of axis 0 of
  ``x`` is expected to be one sample in time. Each element of axis 1 of ``x``
  holds a single data sample. Each element of axis 2 of ``x`` represents the
  measurements of a particular input variable across all times and all data
  items.

- ``targets``: A three-dimensional array of target output data. Each element of
  axis 0 of ``targets`` is expected to be one sample in time. Each element of
  axis 1 of ``targets`` holds a single data sample. Each element of axis 2 of
  ``targets`` represents the measurements of a particular output variable across
  all times and all data items.

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

Unlike a feedforward classifier, where the target labels are provided as a
single vector, a recurrent classifier requires a vector of target labels for
each time step in the input data. So a recurrent classifier model requires the
following inputs for training:

- ``x``: A three-dimensional array of input data. Each element of axis 0 of
  ``x`` is expected to be one sample in time. Each element of axis 1 of ``x``
  holds a single data sample. Each element of axis 2 of ``x`` represents the
  measurements of a particular input variable across all times and all data
  items.

- ``labels``: A two-dimensional array of integer target labels. Each element of
  ``labels`` is expected to be the class index for a single data item. Axis 0 of
  this array represents time, and axis 1 represents data samples.

.. _creating-specifying-layers:

Specifying Layers
=================

One of the most critical bits of creating a neural network model is specifying
how the layers of the network are configured. There are very few limits to the
complexity of possible neural network architectures, so it would be difficult to
create a single, easy-to-use markup that makes it equally easy to specify all
combinations. However, ``theanets`` tries to make it easy to create networks
with a single "stack" of many common types of layers.

When you create a network model, the ``layers`` keyword argument is used to
specify the layers for your network. This keyword argument must be a sequence
of values that specify the configuration of network layers.

Input Layer
-----------

The first element in the ``layers`` tuple should always be an integer; the
:class:`Network.setup_layers() <theanets.feedforward.Network.setup_layers>`
method creates an :class:`Input <theanets.layers.Input>` layer from the first
element in the list.

During training, the input layer can also inject noise into the input data; see
:ref:`training-specifying-regularizers` for more information.

Hidden Layers
-------------

For all hidden layers (i.e., layers that are neither the first nor the last in
the network stack), there are four options for the values of the ``layers``
sequence.

- If a value is an integer, it is interpreted as the size of a vanilla,
  fully-connected feedforward layer. All options for the layer are set to their
  defaults (e.g., the activation for a hidden layer will be given by the
  ``hidden_activation`` network-wide configuration parameter, which defaults to
  a logistic sigmoid).

  For example, to create a network with an input layer containing 4 units,
  hidden layers with 5 and 6 units, and an output layer with 2 units, you can
  just use integers to specify all of your layers::

    net = theanets.Experiment(theanets.Classifier, layers=(4, 5, 6, 2))

- If a value in this sequence is a tuple, it must contain an integer and may
  contain a string. The integer in the tuple specifies the size of the layer. If
  there is a string, and the string names a valid layer type (e.g., ``'tied'``,
  ``'rnn'``, etc.), then this type of layer will be created. Otherwise, the
  string is assumed to name an activation function (e.g., ``'logistic'``,
  ``'relu'``, etc.) and a standard feedforward layer will be created with that
  activation. (See below for a list of predefined activation functions.)

  For example, to create a model with a rectified linear activation in the
  middle layer::

    net = theanets.Classifier(layers=(4, (5, 'relu'), 6))

  Or to create a model with a recurrent middle layer::

    net = theanets.recurrent.Classifier(layers=(4, (5, 'rnn'), 6))

  Note that recurrent models (that is, models containing recurrent layers) are a
  bit different from feedforward ones; please see
  :ref:`creating-recurrent-models` for more details.

- If a value in this sequence is a dictionary, it must contain either a ``size``
  or an ``nout`` key, which specify the number of units in the layer. It can
  additionally contain an ``activation`` key to specify the activation function
  for the layer (see below), and a ``form`` key to specify the type of layer to
  be constructed (e.g., ``'tied'``, ``'rnn'``, etc.). Additional keys in this
  dictionary will be passed as keyword arguments to
  :func:`theanets.layers.build`.

  For example, you can use a dictionary to specify an non-default activation
  function for a layer in your model::

    net = theanets.Regressor(layers=(4, dict(size=5, activation='tanh'), 2))

  You could also create a layer with a sparsely-initialized weight matrix by
  providing the ``sparsity`` key::

    net = theanets.Regressor(layers=(4, dict(size=5, sparsity=0.9), 2))

- Finally, if a value is a :class:`Layer <theanets.layers.Layer>` instance, this
  layer is simply added to the network model as-is.

Output Layer
------------

The output layer in ``theanets`` is the final element of the ``layers`` tuple.
Like the input, this layer must be given as an integer, which specifies the
number of output units in the network. The activation of the output layer is
specified using the ``output_activation`` keyword argument, which defaults to
``'softmax'`` for :class:`classifiers <theanets.feedforward.Classifier>` or
``'linear'`` for :class:`regressors <theanets.feedforward.Regressor>` or
:class:`autoencoder <theanets.feedforward.Autoencoder>` models.

Activation Functions
--------------------

An activation function (sometimes also called a transfer function) specifies how
the output of a layer is computed from the weighted sums of the inputs. By
default, hidden layers in ``theanets`` use a logistic sigmoid activation
function. Output layers in :class:`Regressor <theanets.feedforward.Regressor>`
and :class:`Autoencoder <theanets.feedforward.Autoencoder>` models use linear
activations (i.e., the output is just the weighted sum of the inputs from the
previous layer), and the output layer in :class:`Classifier
<theanets.feedforward.Classifier>` models uses a softmax activation.

To specify a different activation function for a layer, include an activation
key chosen from the table below. As described above, this can be included in
your model specification either using the ``activation`` keyword argument in a
layer dictionary, or by including the key in a tuple with the layer size.

=========  ============================  =============================================
Key        Description                   :math:`g(z) =`
=========  ============================  =============================================
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
rect:max   truncation                    :math:`\min(1, z)`
rect:min   rectification                 :math:`\max(0, z)`
norm:mean  mean-normalization            :math:`z - \bar{z}`
norm:max   max-normalization             :math:`z / \max |z|`
norm:std   variance-normalization        :math:`z / \mathbb{E}[(z-\bar{z})^2]`
=========  ============================  =============================================

.. _creating-customizing:

Customizing
===========

The ``theanets`` package tries to strike a good balance between defining
everything known in the neural networks literature, and allowing you as a
programmer to create new stuff with the library. For many off-the-shelf use
cases, the hope is that something in ``theanets`` will work with just a few
lines of code. For more complex cases, you should be able to create an
appropriate subclass and integrate it into your workflow with a little more
effort.

.. _creating-custom-layers:

Defining Custom Layers
----------------------

Layers are the real workhorse in ``theanets``; custom layers can be created to
do all sorts of fun stuff. To create a custom layer, just subclass :class:`Layer
<theanets.layers.Layer>` and give it the functionality you want. As a very
simple example, let's suppose you wanted to create a normal feedforward layer
but did not want to include a bias term::

  import theanets
  import theano.tensor as TT

  class MyLayer(theanets.layers.Layer):
      def transform(self, inputs):
          return TT.dot(inputs, self.find('w'))

      def setup(self):
          self.log_setup(self.add_weights('w'))

Once you've set up your new layer class, it will automatically be registered and
available in :func:`theanets.layers.build` using the name of your class::

  layer = theanets.layers.build('mylayer', nin=3, nout=4)

or, while creating a model::

  net = theanets.Autoencoder(
      layers=(4, ('mylayer', 'linear', 3), 4),
      tied_weights=True,
  )

This example shows how fast it is to create a model that will learn the subspace
of your dataset that spans the most variance---the same subspace spanned by the
principal components.

.. _creating-custom-regularizers:

Defining Custom Regularizers
----------------------------

To create a custom regularizer in ``theanets``, you need to subclass the
appropriate model and provide an implementation of the
:func:`theanets.feedforward.Network.loss` method.

Let's keep going with the example above. Suppose you created a linear autoencoder
model that had a larger hidden layer than your dataset::

  net = theanets.Autoencoder(layers=(4, ('linear', 8), 4), tied_weights=True)

Then, at least in theory, you risk learning an uninteresting "identity" model
such that some hidden units are never used, and the ones that are have weights
equal to the identity matrix. To prevent this from happening, you can impose a
sparsity penalty::

  net = theanets.Autoencoder(
      layers=(4, ('linear', 8), 4),
      tied_weights=True,
      hidden_l1=0.1,
  )

But then you might run into a situation where the sparsity penalty drives some
of the hidden units in the model to zero, to "save" loss during training.
Zero-valued features are probably not so interesting, so we can introduce
another penalty to prevent feature weights from going to zero::


  class RICA(theanets.Autoencoder):
      def loss(self, **kwargs):
          loss, monitors, updates = super(RICA, self).loss(**kwargs)
          w = kwargs.get('weight_inverse', 0)
          if w > 0:
              loss += w * sum((1 / (p * p).sum(axis=0)).sum()
                              for l in self.layers for p in l.params)
          return loss, monitors, updates

This code adds a new regularizer that penalizes the inverse of the squared
length of each of the weights in the model's layers.

.. _creating-custom-errors:

Defining Custom Error Functions
-------------------------------

It's pretty straightforward to create models in ``theanets`` that use different
error functions from the predefined :class:`Classifier
<theanets.feedforward.Classifier>` (which uses categorical cross-entropy) and
:class:`Autoencoder <theanets.feedforward.Autoencoder>` and :class:`Regressor
<theanets.feedforward.Regressor>` (which both use mean squared error, MSE).

To define by a model with a new cost function, just create a new :class:`Network
<theanets.feedforward.Network>` subclass and override the ``error`` method. For
example, to create a regression model that uses mean absolute error (MAE)
instead of MSE::

  class MaeRegressor(theanets.Regressor):
      def error(self, output):
          return abs(output - self.targets).mean()

Your cost function must return a theano expression that reflects the cost for
your model.
