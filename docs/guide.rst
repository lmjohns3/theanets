.. _guide:

.. rubric:: User Guide

The ``theanets`` package provides tools for defining and optimizing several
common types of neural network models. It uses Python for rapid development, and
under the hood Theano_ provides graph optimization and fast computations on the
GPU. This document describes the high-level ways of using ``theanets``.

.. _Theano: http://deeplearning.net/software/theano/

Installation
============

If you haven't already, the first thing you should do is download and install
``theanets``. The easiest way to do this is by using ``pip``:

.. code:: shell

  pip install theanets

This command will automatically install all of the dependencies for
``theanets``, including ``numpy`` and ``theano``.

If you're feeling adventurous, you can also check out the latest version of
``theanets`` and run the code from your local copy:

.. code:: shell

  git clone https://github.com/lmjohns3/theanets
  cd theanets
  python setup.py develop

This can be risky, however, since ``theanets`` is in active development---the
API might change in the development branch from time to time.

To work through the examples you should also install a couple of supporting
packages:

.. code:: shell

  pip install skdata
  pip install seaborn
  pip install matplotlib

These will help you obtain some common example datasets, and also help in making
plots of various things.

Package Overview
================

At a high level, the ``theanets`` package is a tool for (a) defining and (b)
optimizing cost functions over a set of data. The workflow in ``theanets``
typically involves three basic steps:

#. First, you *define* the structure of the model that you'll use for your task.
   For instance, if you're trying to classify MNIST digits, you'll want
   something that takes in pixels and outputs digit classes (a
   :class:`classifier <theanets.feedforward.Classifier>`). If you're trying to
   model the unlabeled digit images, you might want to use an
   :class:`autoencoder <theanets.feedforward.Autoencoder>`. If you're trying to
   predict the price of a house, say, based on its zip code and size, you might
   want a :class:`regression model <theanets.feedforward.Regressor>`.

#. Second, you *train* or adjust the parameters in your model so that it has a
   low cost or performs well with respect to some task. For classification, you
   might want to adjust your model parameters to minimize the negative
   log-likelihood of the correct image class given the pixels, and for
   autoencoders you might want to minimize the reconstruction error.

#. Finally, you *use* the trained model in some way, probably by predicting
   results on a test dataset, visualizing the learned features, and so on.

If you use ``theanets`` to perform these three steps in one script, the skeleton
of your code will usually look something like this:

.. code:: python

  import theanets

  # 1. create a model -- here, a regression model.
  net = theanets.Regressor([10, 100, 2])

  # optional: set up additional losses.
  net.add_loss('mae', weight=0.1)

  # 2. train the model.
  net.train(
      training_data,
      validation_data,
      algo='rmsprop',
      hidden_l1=0.01,  # apply a regularizer.
  )

  # 3. use the trained model.
  net.predict(test_data)

This user guide describes, at a high level, how to implement these different
stages. Each section links to the relevant API documentation, which provides
more detailed information.

.. _guide-creating:

Creating a Model
================

To use ``theanets``, you will first need to create a neural network model. All
network models in ``theanets`` are instances of the :class:`theanets.Network
<theanets.graph.Network>` base class, which maintains two important pieces of
information:

- a list of :ref:`layers <layers>` that map input data to network outputs, and

- a list of (possibly :ref:`regularized <regularizers>`) :ref:`loss functions
  <losses>` that quantify how well the parameters in the model perform for the
  desired task.

Most of the effort of creating a network model goes into specifying the layers
in the model. We'll take a look at the ways of specifying layers below, and then
talk about how to specify losses and regularizers after that.

.. _guide-creating-specifying-layers:

Specifying Layers
-----------------

Probably the most important part of a neural network model is the
*architecture*---the number and configuration of layers---of the model. There
are very few limits to the complexity of possible neural network architectures,
and ``theanets`` tries to make it possible to create a wide variety of
architectures with minimal effort. The easiest architecture to create, however,
is also the most common: networks with a single "chain" of layers.

For the time being we'll assume that you want to create a regression model with
a single layer chain. To do this, you invoke the constructor of the model class
you wish to create and specify layers you want in the model. For example:

.. code:: python

  net = theanets.Regressor(layers=[10, 20, 3])

Here we've invoked the :class:`theanets.Regressor
<theanets.feedforward.Regressor>` constructor and specified that we want an
input layer with 10 neurons, a hidden layer with 20 neurons, and an output layer
with 3 outputs.

In general, the ``layers`` argument to the constructor must be a sequence of
values, each of which specifies the configuration of a single layer in the
model:

.. code:: python

  net = theanets.Regressor([A, B, ..., Z])

Here, the ``A`` through ``Z`` variables represent layer configuration settings.
As we've seen, these can be plain integers, but if you need to customize one or
more of the layers in your model, you can provide variables of different types.
The different possibilities are discussed below.

Layer Instances
~~~~~~~~~~~~~~~

Any of the values in the layer configuration sequence can be a
:class:`theanets.Layer <theanets.layers.base.Layer>` instance. In this case, the
given layer instance is simply added to the network model as-is.

Integers
~~~~~~~~

If a layer configuration value is an integer, that value is interpreted as the
``size`` of a vanilla :class:`Feedforward
<theanets.layers.feedforward.Feedforward>` layer. All other :ref:`attributes for
the layer <layers-attributes>` are set to their defaults (e.g., the activation
function defaults to "relu").

For example, as we saw above, to create a network with an input layer containing
4 units, hidden layers with 5 and 6 units, and an output layer with 2 units, you
can just use integers to specify all of your layers:

.. code:: python

  net = theanets.Regressor([4, 5, 6, 2])

The :class:`theanets.Network <theanets.graph.Network>` constructor creates
layers for each of these integer values and "connects" them together in a chain
for you.

Tuples
~~~~~~

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
the middle layer and a softmax output layer:

.. code:: python

  net = theanets.Regressor([4, (5, 'sigmoid'), (6, 'softmax')])

Dictionaries
~~~~~~~~~~~~

If a layer configuration value is a dictionary, its keyword arguments are passed
directly to :func:`theanets.Layer.build() <theanets.layers.base.Layer.build>` to
construct a new layer instance.

The dictionary must contain a ``form`` key, which specifies the name of the
layer type to build, as well as a ``size`` key, which specifies the number of
units in the layer. It can additionally contain any other keyword arguments that
you wish to use when constructing the layer.

For example, you can use a dictionary to specify a non-default activation
function for a layer in your model:

.. code:: python

  net = theanets.Regressor([4, dict(size=5, activation='tanh'), 2])

You could also create a layer with a sparsely-initialized weight matrix by
providing the ``sparsity`` key:

.. code:: python

  net = theanets.Regressor([4, dict(size=5, sparsity=0.9), 2])

See the :ref:`attribute documentation <layers-attributes>` for more information
about the keys that can be provided in this dictionary.

Specifying a Loss
-----------------

All of the :ref:`predefined models <models>` in ``theanets`` are created by
default with one :ref:`loss function <losses>` appropriate for that type of
model. You can override or augment this default loss, however, by manipulating
the list of losses, or by providing a non-default loss specifier when creating
your model.

For example, to use a mean-absolute error instead of the default mean-squared
error for a regression model:

.. code:: python

  net = theanets.Regressor([4, 5, 2], loss='mae')

A model can also be trained with multiple losses simultaneously. You can add
losses to your model:

.. code:: python

  net.add_loss('mse', weight=0.1)

Here, the ``weight`` argument specifies the weight of the loss (by default this
is 1). The losses in effect for a model are allowed to change between successive
calls to :func:`theanets.Network.train() <theanets.graph.Network.train>`, so you
can make a model, train it, add a loss, train it more, change the losses, train
a third time, and so on.

See the :ref:`loss documentation <losses>` for more information.

.. _guide-training:

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

.. _training-specifying-trainer:

Specifying a Trainer
--------------------

The easiest way train a model with ``theanets`` is to invoke the :func:`train()
<theanets.graph.Network.train>` method:

.. code:: python

  net = theanets.Classifier(layers=[10, 5, 2])
  net.train(training_data,
            validation_data,
            algo='nag',
            learning_rate=0.01,
            momentum=0.9)

Here, a classifier model is being trained using Nesterov's accelerated gradient,
with a learning rate of 0.01 and momentum of 0.9. You must provide at least a
training dataset, and a validation datasets is a good idea (see below).

The optimization algorithm itself is selected using the ``algo`` keyword
argument, and any other keyword arguments provided to :func:`train()
<theanets.graph.Network.train>` are passed to the algorithm implementation.

Multiple calls to ``train()`` are possible and can be used to implement things
like custom annealing schedules (e.g., the "newbob" training strategy):

.. code:: python

  net = theanets.Classifier(layers=[10, 5, 2])

  for e in (-2, -3, -4):
      net.train(training_data,
                validation_data,
                algo='nag',
                learning_rate=10 ** e,
                momentum=1 - 10 ** (e + 1))

The available training methods are described in the :ref:`trainer documentation
<trainers>`.

.. _guide-training-providing-data:

Providing Data
--------------

To train a model in ``theanets``, you will need to provide a set of data that
can be used to compute the value of the loss function and its derivatives. Data
can be passed to the trainer using either arrays_ or callables_ (this
functionality is provided by the ``downhill`` optimization library).

.. _arrays: http://downhill.rtfd.org/en/stable/guide.html#data-using-arrays
.. _callables: http://downhill.rtfd.org/en/stable/guide.html#data-using-callables

.. _guide-training-specifying-regularizers:

Specifying Regularizers
-----------------------

:ref:`Regularizers <regularizers>` are extra terms added to a model's :ref:`loss
function <losses>` that encourage the model to develop some extra or special
behavior beyond minimizing the loss. Many regularizers are used to prevent model
parameters from growing too large, which is often a sign of overfitting. Other
regularizers are used to encourage a model to develop sparse representations of
the problem space, which can be useful for classification and for human
interpretation of results.

Regularizers in ``theanets`` are specified during training, in calls to
:func:`Network.train() <theanets.graph.Network.train>`, or during use, in calls
to :func:`Network.predict() <theanets.graph.Network.predict>`. Several built-in
regularizers cover the most common cases, but :ref:`custom regularizers
<regularizers-custom>` are fairly easy to implement and use as well.

To specify, for example, that a network model should be trained with weight
decay (that is, using an L2 norm penalty on the weights of the model), just give
the appropriate keyword argument during training:

.. code:: python

  net.train(..., weight_l2=0.01)

Similarly, the hidden representations of a model can be encouraged to be sparse
using the keyword argument:

.. code:: python

  net.train(..., hidden_l1=0.01)

Dropout (multiplicative Bernoulli noise) and additive (Gaussian) noise are also
common regularization techniques. Like other regularizers, these can be applied
during training and/or use. For example, to apply dropout to the input layer
when predicting a sample:

.. code:: python

  predicted = net.predict(sample, input_dropout=0.1)

or, for example, to apply noise to the hidden representations during training:

.. code:: python

  net.train(..., hidden_noise=0.1)

See the :ref:`regularizer documentation <regularizers>` for more information.

.. _guide-training-iteration:

Training as Iteration
---------------------

The :func:`theanets.Network.train() <theanets.graph.Network.train>` method is
actually just a thin wrapper over the underlying
:func:`theanets.Network.itertrain() <theanets.graph.Network.itertrain>` method,
which you can use directly if you want to do something special during training:

.. code:: python

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

The :class:`theanets.Network <theanets.graph.Network>` base class can snapshot
your model automatically during training. When you call
:func:`theanets.Network.train() <theanets.graph.Network.train>`, you can provide
the following keyword arguments:

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

You can also save and load models manually by calling
:func:`theanets.Network.save() <theanets.graph.Network.save>` and
:func:`theanets.Network.load() <theanets.graph.Network.load>`, respectively.

.. _guide-using:

Using a Model
=============

Once you've trained a model, you will probably want to do something useful with
it. If you are working in a production environment, you might want to use the
model to make predictions about incoming data; if you are doing research, you
might want to examine the parameters that the model has learned.

Predicting New Data
-------------------

For most neural network models, you can compute the "natural" output of the
model layer by calling :func:`theanets.Network.predict()
<theanets.graph.Network.predict>`:

.. code:: python

  results = net.predict(new_dataset)

For :class:`regression <theanets.feedforward.Regressor>` and
:class:`autoencoding <theanets.feedforward.Autoencoder>` models, this method
returns the output of the network when passed the given input dataset. For
:class:`classification <theanets.feedforward.Classifier>` models, this method
returns the predicted classification of the inputs. (To get the actual output of
the network---the posterior class probabilities---for a classifier model, use
:func:`predict_proba() <theanets.feedforward.Classifier.predict_proba>`.)

Regardless of the model, you pass to ``predict()`` a ``numpy`` array containing
data examples along the rows, and the method returns an array containing one row
of output predictions for each row of input data.

You can also compute the activations of all layer outputs in the network using
the :func:`theanets.Network.feed_forward()
<theanets.feedforward.Network.feed_forward>` method:

.. code:: python

  for name, value in net.feed_forward(new_dataset).items():
      print(abs(value).sum(axis=1))

This method returns a dictionary that maps layer output names to their
corresponding values for the given input. Like ``predict()``, each output array
contains one row for every row of input data.

Inspecting Parameters
---------------------

The parameters in each layer of the model are available using
:func:`theanets.Network.find() <theanets.feedforward.Network.find>`. This method
takes two query terms---either integer index values or string names---and
returns a theano shared variable for the given parameter. The first query term
finds a layer in the network, and the second finds a parameter within that
layer.

The ``find()`` method returns a `Theano shared variable`_. To get a numpy array
of the current values of the variable, call ``get_value()`` on the result from
``find()``, like so:

.. code:: python

  param = net.find('hid1', 'w')
  values = param.get_value()

For "encoding" layers in the network, this value array contains a feature vector
in each column, and for "decoding" layers (i.e., layers connected to the output
of an autoencoder), the features are in each row.

.. _Theano shared variable: http://deeplearning.net/software/theano/library/compile/shared.html

Visualizing Weights
-------------------

Many times it is useful to create a plot of the features that the model learns;
this can be useful for debugging model performance, but also for interpreting
the dataset through the "lens" of the learned features.

For example, if you have a model that takes as input a 28×28 MNIST digit, then
you could plot the weight vectors attached to each unit in the first hidden
layer of the model to see what sorts of features the hidden unit detects:

.. code:: python

  img = np.zeros((28 * 10, 28 * 10), dtype='f')
  for i, pix in enumerate(net.find('hid1', 'w').get_value().T):
      r, c = divmod(i, 10)
      img[r * 28:(r+1) * 28, c * 28:(c+1) * 28] = pix.reshape((28, 28))
  plt.imshow(img, cmap=plt.cm.gray)
  plt.show()

Here we've taken the weights from the first hidden layer of the model
(``net.find('hid1', 'w')``) and plotted them as though they were 28×28 grayscale
images. This is a useful technique for processing images (and, to some extent,
other types of data) because visually inspecting features can give you a quick
sense of how the model interprets its input. In addition, this can serve as a
sanity check---if the features in the model look like TV snow, for example, the
model probably hasn't adapted its weights properly, so something might be wrong
with the training process.

.. _guide-advanced:

Advanced Topics
===============

.. _guide-advanced-customizing:

Customizing
-----------

The ``theanets`` package tries to strike a balance between defining everything
known in the neural networks literature, and allowing you as a programmer to
create new and exciting stuff with the library. For many off-the-shelf use
cases, the hope is that something in ``theanets`` will work with just a few
lines of code. For more complex cases, you should be able to create an
appropriate subclass and integrate it into your workflow with just a little more
effort.

Nearly every major base class in ``theanets`` can be subclassed and applied
directly in your model:

- :class:`theanets.Layer <theanets.layers.base.Layer>` --- see :ref:`layers-custom`
- :class:`theanets.Activation <theanets.activations.Activation>` --- see :ref:`activations-custom`
- :class:`theanets.Loss <theanets.losses.Loss>` --- see :ref:`losses-custom`
- :class:`theanets.Regularizer <theanets.regularizers.Regularizer>` --- see :ref:`regularizers-custom`
- :class:`theanets.Network <theanets.graph.Network>` --- see :ref:`models-custom`

These classes form the bulk of the ``theanets`` framework; understanding how
they can be customized gives almost all of the flexibility that ``theanets``
provides, particularly when combined with the ability to create arbitrary
computation graphs.

.. _guide-advanced-graphs:

Computation Graphs
------------------

While many types of neural networks are constructed using a single linear
"chain" of layers, this does not always need to be the case. Indeed, many of the
more exotic model types that perform well in specialized settings make use of
connections between multiple inputs and outputs.

In ``theanets`` it is easiest to create network architectures that use a single
chain of layers. However, it is also possible to create network graphs that have
arbitrary, acyclic connections among layers. Creating a nonlinear network graph
requires using the ``inputs`` keyword argument when creating a layer.

The ``inputs`` keyword argument for creating a layer should be a list of strings
that specifies the names of one or more network outputs. If ``inputs`` is not
specified for a layer, ``theanets`` creates a default input specification that
uses the output from the previous layer.

Perhaps the simplest example of a non-default ``inputs`` specification is to
create a classifier model that uses outputs from all hidden layers to inform the
final output of the layer. Such a "multi-scale" model can be created as follows:

.. code:: python

  theanets.Classifier((
      784,
      dict(size=100, name='a'),
      dict(size=100, name='b'),
      dict(size=100, name='c'),
      dict(size=10, inputs=('a', 'b', 'c')),
  ))

Here, each of the hidden layers is assigned an explicit name, so that they will
be easy to reference by the last layer. The output layer, a vanilla feedforward
layer, combines together the outputs from layers ``a``, ``b``, and ``c``.

More Information
================

This concludes the user guide! Please read more information about ``theanets``
in the `examples and API documentation`_.

.. _examples and API documentation: ./

The source code for ``theanets`` lives at http://github.com/lmjohns3/theanets.
Please fork, explore, and send pull requests!

Finally, there is also a mailing list for project discussion and announcements.
Subscribe online at https://groups.google.com/forum/#!forum/theanets.
