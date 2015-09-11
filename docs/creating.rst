.. _creating:

==================
Creating a Network
==================

To use ``theanets``, you will first need to create a neural network model. All
network models in ``theanets`` are instances of the :class:`Network
<theanets.graph.Network>` base class, which maintains two important pieces of
information:

- a list of :class:`Layers <theanets.layers.base.Layer>` that map input data to
  network outputs, and
- a list of (possibly :class:`regularized <theanets.regularizers.Regularizer>`)
  :class:`loss functions <theanets.losses.Loss>` that quantify how well the
  parameters in the model perform for the desired task.

Most of the effort of creating a network model goes into specifying the layers
in the model. We'll take a look at the ways of specifying layers below, and then
talk about how to specify losses and regularizers after that.

.. _creating-specifying-layers:

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

Layers
------

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

.. _creating-custom-activations:

Activations
-----------

.. _creating-custom-losses:

Losses
------

It's pretty straightforward to create models in ``theanets`` that use different
losses from the predefined :class:`Classifier <theanets.feedforward.Classifier>`
and :class:`Autoencoder <theanets.feedforward.Autoencoder>` and
:class:`Regressor <theanets.feedforward.Regressor>` models. (The classifier uses
categorical cross-entropy (XE) as its default loss, and the other two both use
mean squared error, MSE.)

To define a model with a new loss, just create a new :class:`Loss
<theanets.losses.Loss>` subclass and specify its name when you create your
model. For example, to create a regression model that uses a step function
averaged over all of the model inputs::

  class Step(theanets.Loss):
      def __call__(self, outputs):
          return (outputs[self.output_name] > 0).mean()

  net = theanets.Regressor([5, 6, 7], loss='step')

Your loss function implementation must return a Theano expression that reflects
the loss for your model. If you wish to make your loss work with weighted
outputs, you will also need to include a case for having weights::

  class Step(theanets.Loss):
      def __call__(self, outputs):
          step = outputs[self.output_name] > 0
          if self._weights:
              return (self._weights * step).sum() / self._weights.sum()
          else:
              return step.mean()

.. _creating-custom-regularizers:

Regularizers
------------

To create a custom regularizer in ``theanets``, you need to create a custom
subclass of the :class:`Regularizer <theanets.regularizers.Regularizer>` class,
and then provide this regularizer when you run your model.

To illustrate, let's suppose you created a linear autoencoder model that had a
larger hidden layer than your dataset::

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

.. _creating-graphs:

Creating Graphs
===============

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
