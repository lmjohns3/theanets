.. _guide:

.. rubric:: User Guide

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

.. _guide-creating:

Creating a Model
================

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

.. _guide-training-providing-data:

Providing Data
--------------

To train a model in ``theanets``, you will need to provide a set of data that
can be used to compute the value of the loss function and its derivatives. Data
can be passed to the trainer using either arrays_ or callables_; the
``downhill`` documentation describes how this works.

.. _arrays: http://downhill.rtfd.org/en/stable/guide.html#data-using-arrays
.. _callables: http://downhill.rtfd.org/en/stable/guide.html#data-using-callables

.. _guide-training-iteration:

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

.. _guide-using:

Using a Model
=============

Once you've trained a model, you will probably want to do something useful with
it. If you are working in a production environment, you might want to use the
model to make predictions about incoming data; if you are doing research, you
might want to examine the parameters that the model has learned.

Predicting New Data
-------------------

For most neural network models, you can compute the "natural" output of the model
layer by calling :func:`Network.predict() <theanets.graph.Network.predict>`::

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
the :func:`Network.feed_forward() <theanets.feedforward.Network.feed_forward>`
method::

  for name, value in net.feed_forward(new_dataset).items():
      print(abs(value).sum(axis=1))

This method returns a dictionary that maps layer output names to their
corresponding values for the given input. Like ``predict()``, each output array
contains one row for every row of input data.

Inspecting Learned Parameters
-----------------------------

The parameters in each layer of the model are available using
:func:`Network.find() <theanets.feedforward.Network.find>`. This method takes
two query terms---either integer index values or string names---and returns a
theano shared variable for the given parameter. The first query term finds a
layer in the network, and the second finds a parameter within that layer.

The ``find()`` method returns a `Theano shared variable`_. To get a numpy array
of the current values of the variable, call ``get_value()`` on the result from
``find()``, like so::

  values = net.find('hid1', 'w').get_value()

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
layer of the model to see what sorts of features the hidden unit detects::

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

.. _creating-graphs:

Computation Graphs
------------------

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

.. _guide-customizing:

Customizing
-----------

The ``theanets`` package tries to strike a balance between defining everything
known in the neural networks literature, and allowing you as a programmer to
create new and exciting stuff with the library. For many off-the-shelf use
cases, the hope is that something in ``theanets`` will work with just a few
lines of code. For more complex cases, you should be able to create an
appropriate subclass and integrate it into your workflow with just a little more
effort.

