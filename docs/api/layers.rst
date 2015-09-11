.. _layers:

======
Layers
======

Probably the most important part of a neural network model is the
*architecture*---the number and configuration of layers---of the model. There
are very few limits to the complexity of possible neural network architectures,
and ``theanets`` tries to make it possible to create a wide variety of
architectures with minimal effort. The easiest architecture to create, however,
is also the most common: networks with a single "chain" of layers.

For the time being we'll assume that you want to create a regression model with
a single layer chain. To do this, you invoke the constructor of the model class
you wish to create and specify layers you want in the model. For example::

  net = theanets.Regressor(layers=[10, 20, 3])

Here we've invoked the :class:`theanets.Regressor
<theanets.feedforward.Regressor>` constructor and specified that we want an
input layer with 10 neurons, a hidden layer with 20 neurons, and an output layer
with 3 outputs.

Specifying Layers
=================

In general, the ``layers`` argument to the constructor must be a sequence of
values, each of which specifies the configuration of a single layer in the
model::

  net = theanets.Regressor([A, B, ..., Z])

Here, the ``A`` through ``Z`` variables represent layer configuration settings.
As we've seen, these can be plain integers, but if you need to customize one or
more of the layers in your model, you can provide variables of different types.
The different possibilities are discussed below.

Layer Instances
---------------

Any of the values in the layer configuration sequence can be a :class:`Layer
<theanets.layers.base.Layer>` instance. In this case, the given layer instance
is simply added to the network model as-is.

Integers
--------

If a layer configuration value is an integer, that value is interpreted as the
``size`` of a vanilla :class:`Feedforward
<theanets.layers.feedforward.Feedforward>` layer. All other attributes for the
layer---summarized below in :ref:`creating-default-attributes`---are set to
their defaults (e.g., the activation function defaults to "relu").

For example, as we saw above, to create a network with an input layer containing
4 units, hidden layers with 5 and 6 units, and an output layer with 2 units, you
can just use integers to specify all of your layers::

  net = theanets.Regressor([4, 5, 6, 2])

The :class:`Network <theanets.graph.Network>` constructor creates layers for
each of these integer values and "connects" them together in a chain for you.

Tuples
------

Sometimes you will want to specify more than just the size of a layer. Commonly,
modelers want to change the "form" (i.e., the type of the layer), or its
activation function. A tuple is a good way to specify these attributes. If a
layer configuration value is a tuple, it must contain an integer and may contain
one or more strings.

The integer in the tuple specifies the ``size`` of the layer.

If there is a string in the tuple that names a registered layer type (e.g.,
``'tied'``, ``'rnn'``, etc.), then this type of layer will be created.

If there is a string in the tuple and it does not name a registered layer type,
the string is assumed to name an activation function---for example,
``'logistic'``, ``'relu+norm:z'``, and so on.

For example, to create a regression model with a logistic sigmoid activation in
the middle layer and a softmax output layer::

  net = theanets.Regressor([4, (5, 'sigmoid'), (6, 'softmax')])

Dictionaries
------------

If a layer configuration value is a dictionary, its keyword arguments are passed
directly to :func:`theanets.Layer.build() <theanets.util.Registrar.build>` to
construct a new layer instance.

The dictionary must contain a ``form`` key, which specifies the name of the
layer type to build, as well as a ``size`` key, which specifies the number of
units in the layer. It can additionally contain any other keyword arguments that
you wish to use when constructing the layer.

For example, you can use a dictionary to specify a non-default activation
function for a layer in your model::

  net = theanets.Regressor([4, dict(size=5, activation='tanh'), 2])

You could also create a layer with a sparsely-initialized weight matrix by
providing the ``sparsity`` key::

  net = theanets.Regressor([4, dict(size=5, sparsity=0.9), 2])

.. _layers-attributes:

Layer Attributes
================

Now that we've seen how to specify values for the attributes of each layer in
your model, we'll look at the available attributes that can be customized. For
many of these settings, you'll want to use a dictionary (or create a
:class:`Layer <theanets.layers.base.Layer>` instance yourself) to specify
non-default values.

- ``size``: The number of "neurons" in the layer. This value must be specified
  by the modeler when creating the layer. It can be specified by providing an
  integer, or as a tuple that contains an integer.

- ``form``: A string specifying the :ref:`type of layer <creating-layer-types>`
  to use. This defaults to "feedforward" but can be the name of any existing
  :class:`Layer <theanets.layers.base.Layer>` subclass (including :ref:`custom
  layers <creating-custom-layers>` that you have defined).

- ``name``: A string name for the layer. If this isn't provided when creating a
  layer, the layer will be assigned a default name. The default names for the
  first and last layers in a network are ``'in'`` and ``'out'`` respectively,
  and the layers in between are assigned the name "hidN" where N is the number
  of existing layers.

  If you create a layer instance manually, the default name is ``'layerN'``
  where N is the number of existing layers.

- ``activation``: A string describing the :ref:`activation function
  <creating-activation>` to use for the layer. This defaults to ``'relu'``.

- ``inputs``: An integer or dictionary describing the sizes of the inputs that
  this layer expects. This is normally optional and defaults to the size of the
  preceding layer in a chain-like model. However, providing a dictionary here
  permits arbitrary layer interconnections. See :ref:`creating-graphs` for more
  details.

- ``mean``: A float specifying the mean of the initial parameter values to use
  in the layer. Defaults to 0. This value applies to all parameters in the model
  that don't have mean values specified for them directly.

- ``mean_ABC``: A float specifying the mean of the initial parameter values to
  use in the layer's ``'ABC'`` parameter. Defaults to 0. This can be used to
  specify the mean of the initial values used for a specific parameter in the
  model.

- ``std``: A float specifying the standard deviation of the initial parameter
  values to use in the layer. Defaults to 1. This value applies to all
  parameters in the model that don't have standard deviations specified
  directly.

- ``std_ABC``: A float specifying the standard deviation of the initial
  parameter values to use in the layer's ``'ABC'`` parameter. Defaults to 1.
  This can be used to specify the standard deviation of the initial values used
  for a specific parameter in the model.

- ``sparsity``: A float giving the proportion of parameter values in the layer
  that should be initialized to zero. Nonzero values in the parameters will be
  drawn from a Gaussian with the specified mean and standard deviation as above,
  and then an appropriate number of these parameter values will randomly be
  reset to zero to make the parameter "sparse."

- ``sparsity_ABC``: A float or vector of floats used to initialize the
  parameters in the layer's ``'ABC'`` parameter. This can be used to set the
  initial sparsity level for a particular parameter in the layer.

- ``diagonal``: A float or vector of floats used to initialize the parameters in
  the layer. If this is provided, weight matrices in the layer will be
  initialized to all zeros, with this value or values placed along the diagonal.

- ``diagonal_ABC``: A float or vector of floats used to initialize the
  parameters in the layer's ``'ABC'`` parameter. If this is provided, the
  relevant weight matrix in the layer will be initialized to all zeros, with
  this value or values placed along the diagonal.

- ``rng``: An integer or ``numpy`` random number generator. If specified the
  given random number generator will be used to create the initial values for
  the parameters in the layer. This can be useful for repeatable runs of a
  model.

In addition to these configuration values, each layer can also be provided with
keyword arguments specific to that layer. For example, the :class:`MRNN
<theanets.layers.recurrent.MRNN>` recurrent layer type requires a ``factors``
argument, and the :class:`Conv1 <theanets.layers.convolution.Conv1>` 1D
convolutional layer requires a ``filter_size`` argument.

.. _layers-predefined:

Predefined Layers
=================

There are many types of layers available out of the box in ``theanets``.

Input
-----

:Key: :class:`input <theanets.layers.base.Input>`
:Parameters:
:Outputs: out
:Arguments: ``ndim``

Input layers are responsible for the Theano variables that represent input to a
network. The name of the layer is passed along to the symbolic Theano input
variable.

Input layers accept an ``ndim`` argument that specifies the number of dimensions
required to hold mini-batches of the input data. This defaults to 2.

Feedforward
-----------

:Key: :class:`feedforward <theanets.layers.feedforward.Feedforward>`
:Parameters: b w (with one input), b w_1 w_2 ... w_N (with N inputs)
:Outputs: out pre

The vanilla feedforward layer computes a weighted sum of its inputs.

:Key: :class:`classifier <theanets.layers.feedforward.Classifier>`
:Parameters:
:Outputs: out pre

The classifier layer is just a vanilla feedforward layer that uses a softmax
output activation.

:Key: :class:`tied <theanets.layers.feedforward.Tied>`
:Parameters: b
:Outputs: out pre
:Arguments: ``partner``

A "tied" layer is a feedforward layer that uses the transposed weight matrix
from a ``partner`` layer, which can be specified as a string inside a layers
list, or as a direct reference to the partner layer.

Often this type of layer is used in autoencoder models to reduce the number of
parameters.

Recurrent
---------

Recurrent layers must be used with :mod:`recurrent models <theanets.recurrent>`.
They represent layers that incorporate the layer's state from previous time
steps.

:Key: :class:`rnn <theanets.layers.recurrent.RNN>`
:Parameters: b xh hh
:Outputs: out pre

A vanilla recurrent layer.

:Key: :class:`arrnn <theanets.layers.recurrent.ARRNN>`
:Parameters: b r xh xr hh
:Outputs: out pre hid rate

A recurrent layer that has an "adaptive" rate parameter for each neuron in the
layer. The rates are adaptive because they are computed based on the current
input to the network.

This layer type is between the :class:`GRU <theanets.layers.recurrent.GRU>` and
the :class:`LRRNN <theanets.layers.recurrent.LRRNN>` in complexity.

:Key: :class:`lrrnn <theanets.layers.recurrent.LRRNN>`
:Parameters: b r xh hh
:Outputs: out pre hid rate

A recurrent layer that has a "learned" rate parameter for each neuron in the
layer. The vector of rates is learnable but independent of the hidden state and
the input to the network.

The :class:`LRRNN <theanets.layers.recurrent.LRRNN>` is a more complex version
of this layer.

:Key: :class:`lstm <theanets.layers.recurrent.LSTM>`
:Parameters: b ci cf co xh hh
:Outputs: out cell

A Long Short-Term Memory (LSTM) layer is a complex arrangement of parameters
with several dedicated "gates" that permit information to flow into and out of
the "cell" that each neuron represents.

:Key: :class:`mrnn <theanets.layers.recurrent.MRNN>`
:Parameters: b xh xf hf fh
:Outputs: out pre factors

A Multiplicative RNN factors the hidden dynamics of a vanilla RNN into a product
of two matrices. Often this factored representation is a lower rank than the
full dynamics. Furthermore, the factor activations of the hidden dynamics are
modulated by the input to the network at each time step.

:Key: :class:`mut1 <theanets.layers.recurrent.MUT1>`
:Parameters: b xh xr xz hh hr bh br bz
:Outputs: out pre

:Key: :class:`gru <theanets.layers.recurrent.GRU>`
:Parameters: b xh xr xz hh hr hz bh br bz
:Outputs: out pre hid rate

:Key: :class:`clockwork <theanets.layers.recurrent.Clockwork>`
:Parameters: b xh hh
:Outputs: out pre
:Arguments: ``periods``

:Key: :class:`bidirectional <theanets.layers.recurrent.Bidirectional>`
:Outputs: out pre fw_XYZ bw_XYZ
:Arguments: ``worker``

Graph
-----

Several ``theanets`` layers manipulate data for further processing in the graph.
None of these layer types applies an activation function.

:Key: :class:`product <theanets.layers.base.Product>`
:Outputs: out

This layer performs an elementwise multiplication of multiple inputs; all inputs
must be the same shape.

:Key: :class:`concatenate <theanets.layers.base.Concatenate>`
:Outputs: out

This layer concatenates multiple inputs along their last dimension; all inputs
must have the same dimensionality and the same shape along all but the last
dimension. The size of this layer must equal the sum of the sizes of the inputs.

:Key: :class:`flatten <theanets.layers.base.Flatten>`
:Outputs: out

This layer flattens its inputs along all but the first dimension, so that the
layer always outputs an array of dimension 2. The ``size`` value must be correct
for this layer, equal to the product of the shapes of its input!

:Key: :class:`reshape <theanets.layers.base.Reshape>`
:Outputs: out
:Arguments: ``shape``

This layer reshapes its input along all but the first dimension to a new shape.
The shape must be consistent with the shape of the input array.

.. _layers-custom:

Custom Layers
=============

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
          self.add_weights('w', nin=self.input_size, nout=self.size)

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

