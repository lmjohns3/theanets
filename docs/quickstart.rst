===============
Getting Started
===============

This page provides a quick overview of the ``theanets`` package. It is aimed at
getting you up and running with a few simple examples. Once you understand the
basic workflow, you will be able to extend the examples to your own datasets and
modeling problems.

.. _qs-setup:

Installation
============

First of all, you'll probably want to download and install ``theanets``. The
easiest way to do this is by using ``pip``::

  pip install theanets

This command will automatically install all of the dependencies for
``theanets``, including ``numpy`` and ``theano``.

To work through the documentation you should also install a couple of supporting
packages::

  pip install skdata
  pip install matplotlib

These will help you obtain the example dataset described below, and also help in
making plots of various things.

.. _qs-overview:

Package Overview
================

At a high level, the ``theanets`` package is a tool for (a) defining and (b)
optimizing cost functions over a set of data. The workflow in ``theanets``
typically involves three basic steps:

#. First, you define the structure of the model that you'll use for your task.
   For instance, if you're trying to classify MNIST digits, then you'll want
   something that takes in pixels and outputs digit classes (a "classifier"). If
   you're trying to model the digit images without labels, you might want to use
   a model that takes in pixels and outputs pixels (an "autoencoder").
#. Second, you train or adjust the parameters in your model so that it has a low
   cost or performs well with respect to some task. For classification, you
   might want to adjust your model parameters to minimize the negative
   log-likelihood of the correct image class given the pixels, and for
   autoencoders you might want to minimize the reconstruction error.
#. Finally, you use the trained model in some way, probably by predicting
   results on a test dataset, visualizing the learned features, and so on.

The ``theanets`` package provides a helper class, :class:`Experiment
<theanets.main.Experiment>`, that performs these tasks with relatively low
effort on your part. You will typically define a model by creating an experiment
with a number of *model hyperparameters* that define the specific behavior of
your model, and then train your model using another set of *optimization
hyperparameters* that define the behavior of the optimization algorithm.

The skeleton of your code will usually look something like this::

  import matplotlib.pyplot as plt
  import skdata.mnist
  import theanets

  # create an experiment to define and train a model.
  exp = theanets.Experiment(
      Model,
      hyperparam1=value1,
      hyperparam2=value2,
      # ...
  )

  # train the model.
  exp.train(
      training_data,
      validation_data,
      optimize='foo',
      # ...
  )

  # use the trained model.
  model = exp.network
  model.predict(test_data)

This quickstart document shows how to implement these stages by following a
couple of examples below.

.. _qs-classifier:

Classifying MNIST Digits
========================

Suppose you're interested in learning a model that can classify an image of an
MNIST digit as a 0, a 1, a 2, etc. For this task, you would normally use the
:class:`Classifier <theanets.feedforward.Classifier>` feedforward network model.

Classifier networks map a layer of continuous-valued inputs through one or more
hidden layers and finally to an output layer that is activated through the
`softmax function`_. The softmax output is treated as a categorical distribution
over the digit labels given the input image.

The first ("input") and last ("output") layers in your network must match the
size of the data you'll be providing. For an MNIST classification task, this
means your network must have 784 inputs (one for each image pixel) and 10
outputs (one for each class).

Classifier models can be constructed with any number of layers between the input
and output. Models with more than about two hidden layers are commonly called
"deep" models and have been quite popular recently due to their success on a
variety of difficult machine learning problems.

.. _softmax function: http://en.wikipedia.org/wiki/Softmax_function

Defining the model
------------------

Having chosen a model class to use for your task, and a set of layer sizes that
you want in your model, you will create an :class:`Experiment
<theanets.main.Experiment>` to construct your model.

There are two required arguments: the class of the model to create, and the
``layers`` keyword argument, which specifies the number and size of the layers
in your network.  define a classifier model::

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(784, 100, 10))

This is all you need to do to define a classifier model that can be trained up
and used. There are many more hyperparameters available, but for now we'll stick
with the defaults.

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

.. _qs-training:

Training a Model
================

So far, the code above is sufficient to instruct ``theanets`` to create a model.
But models are initialized using small random values for the parameters, which
are unlikely to do anything useful with an MNIST digit as input! To improve the
performance of a model, you'll need to *train* or *optimize* it by adjusting the
model parameters.

The :class:`Experiment <theanets.main.Experiment>` class handles the general
case of training with fairly little work. Most of the effort required here is in
processing your dataset so that you can use it to train a network.

Preparing a dataset
-------------------

.. _qs-mnist:

MNIST digits
------------

The examples throughout the documentation use the `MNIST digits dataset
<http://yann.lecun.com/exdb/mnist/>`_, a set of 70,000 28×28 images of
hand-written digits. Each MNIST digit is labeled with the correct digit class
(0, 1, ... 9).

.. image:: http://www.heikohoffmann.de/htmlthesis/img679.gif

Because the MNIST digits are high-dimensional pixel arrays, they are useful for
evaluating models of unsupervised learning like autoencoders. But because the
MNIST digits are also labeled, they are also useful for evaluating models of
supervised learning like classifiers. We'll address both of these tasks as
examples in this document.

For now, you can look at a few of the digits in the image above, or by plotting
them on your computer::

  import matplotlib.pyplot as plt
  import numpy.random as rng
  import skdata.mnist

  mnist = skdata.mnist.dataset.MNIST()
  mnist.meta  # trigger download if needed.
  digits = mnist.arrays['train_images']

  # show a 5 x 5 grid of MNIST samples.
  for axes in plt.subplots(5, 5)[1]:
      for ax in axes:
          ax.imshow(digits[rng.randint(len(digits))])
          ax.set_xticks([])
          ax.set_yticks([])
          ax.set_frame_on(False)

  plt.show()

Before you can train your model, you'll need to write a little glue code to
arrange for a training and a validation dataset. With the MNIST digits, this is
pretty straightforward::

  def load_mnist():
      mnist = skdata.mnist.dataset.MNIST()
      mnist.meta  # trigger download if needed.
      def arr(n, dtype):
          # convert an array to the proper shape and dtype
          arr = mnist.arrays[n]
          return arr.reshape((len(arr), -1)).astype(dtype)
      train_images = arr('train_images', 'f') / 255.
      train_labels = arr('train_labels', np.uint8)
      test_images = arr('test_images', 'f') / 255.
      test_labels = arr('test_labels', np.uint8)
      return ((train_images[:50000], train_labels[:50000, 0]),
              (train_images[50000:], train_labels[50000:, 0]),
              (test_images, test_labels[:, 0]))

Here we've rescaled the image data so that each pixel lies in the interval
:math:`[0, 1]` instead of the default :math:`[0, 255]`. We've also reshaped the
data so that each example is a 1-dimensional vector rather than a 2-dimensional
array of pixels.

The load function returns a training split (the first 50000 examples), a
validation split (the remainder of the training data from ``skdata``, containing
10000 examples), and a test split (the test split from ``skdata``, containing
10000 examples).

.. note::

   Because ``theanets`` uses Theano for its computations, most datasets need to
   be cast to a value that is compatible with your setting for
   `Theano's "floatX" configuration parameter`_. Unless you have a really
   expensive GPU, this is likely to mean that you need to use 32-bit floats.

.. _Theano's "floatX" configuration parameter: http://deeplearning.net/software/theano/library/config.html#config.floatX

Choosing an optimizer
---------------------

The next step is to specify the training algorithm to use, and any associated
hyperparameter values. This is most naturally accomplished using the
:func:`train() <theanets.main.Experiment.train>` method of the experiment
object::

  exp.train(training_data,
            optimize='nag',
            learning_rate=1e-3,
            momentum=0.9)

The first positional argument to this method is the training dataset, and the
second (if provided) is a validation dataset. (These positional arguments can
also be passed to :func:`Experiment.train() <theanets.main.Experiment.train>`
using the keywords ``train_set`` and ``valid_set``, respectively.) If a
validation dataset is not provided, the training dataset will be used for
validation.

The ``optimize`` keyword argument specifies an algorithm to use for training.
(If you do not provide a value for this argument, ``'rmsprop'`` is used by
default, using the :class:`RmsProp <theanets.trainer.RmsProp>` algorithm.) Any
subsequent keyword arguments will be passed to the training algorithm
implementation; these arguments typically specify hyperparameters of the
training algorithm like the learning rate and so forth.

The available training methods are described mathematically in :doc:`trainers`;
here we've specified :class:`Nesterov's Accelerated Gradient
<theanets.trainer.NAG>`, a type of stochastic gradient descent with momentum.

To train our model, we will use the MNIST digits dataset from above. Putting
everything together yields code that looks like this::

  train, valid, _ = load_mnist_labeled()
  exp = theanets.Experiment(theanets.Classifier, layers=(784, 100, 10))
  exp.train(train, valid, optimize='nag', learning_rate=1e-3, momentum=0.9)

If you put this code, plus any necessary imports, into a file called something
like ``mnist-classifier.py``, and then run it on the command-line, your computer
will do a bunch of work to learn good parameter values for your model! If you
enable Python's ``logging`` module you'll also get updates on the console about
the progress of the optimization procedure.

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

Saving and loading
------------------

The :class:`Experiment <theanets.main.Experiment>` class can snapshot your model
automatically during training. When you call :func:`Experiment.train()
<theanets.main.Experiment.train>`, you can provide the following keyword
arguments:

- ``save_progress``: This should be a string containing a filename where the
  model should be saved.

- ``save_every``: This should be a numeric value specifying how often the model
  should be saved during training. If this value is positive, it specifies the
  number of training iterations between checkpoints; if it is negative, it
  specifies the number of minutes that are allowed to elapse between
  checkpoints.

If you provide a ``save_progress`` argument when you construct your experiment,
and a model exists in the given snapshot file, then that model will be loaded
from disk.

You can also save and load models manually by calling :func:`Experiment.save()
<theanets.main.Experiment.save>` and :func:`Experiment.load()
<theanets.main.Experiment.load>`, respectively.

.. _qs-using:

Using a Model
=============

Once you've trained a model, you will probably want to do something useful with
it. If you are working in a production environment, you might want to use the
model to make predictions about incoming data; if you are doing research, you
might want to examine the parameters that the model has learned.

Computing feedforward activations
---------------------------------

For all neural network models, you can compute the activation of the output
layer by calling :func:`Network.predict()
<theanets.feedforward.Network.predict>`::

  results = exp.network.predict(new_dataset)

This returns an array containing one row of output activations for each row of
input data. You can also compute the activations of all layers in the network
using the :func:`Network.feed_forward()
<theanets.feedforward.Network.feed_forward>` method::

  for layer in exp.network.feed_forward(new_dataset):
      print(abs(layer).sum(axis=1))

This method returns a list of arrays, one for each layer in the network. Each
array contains one row for every row of input data.

Additionally, for classifiers, you can obtain predictions for new data using the
:func:`Classifier.classify() <theanets.feedforward.Classifier.classify>`
method::

  classes = exp.network.classify(new_dataset)

This returns a vector of integers; each element in the vector gives the greedy
(argmax) result across the categories for the corresponding row of input data.

Visualizing learned features
----------------------------

Many times it is useful to create a plot of the features that the model learns;
this can be useful for debugging model performance, but also for interpreting
the dataset through the "lens" of the learned features.

The weights connecting successive layers of neurons in the model are available
using :func:`Network.get_weights() <theanets.feedforward.Network.get_weights>`.
This method takes an integer, the index of the weights to retrieve, and returns
an array containing the weights. For "encoding" layers in the network, this
array contains a feature vector in each column (for "decoding" layers, the
features are in each row).

For a dataset like the MNIST digits, you can reshape the learned features and
visualize them as though they were 28×28 images::

  img = np.zeros((28 * 10, 28 * 10), dtype='f')
  for i, pix in enumerate(exp.network.get_weights(0).T):
      r, c = divmod(i, 10)
      img[r * 28:(r+1) * 28, c * 28:(c+1) * 28] = pix.reshape((28, 28))
  plt.imshow(img, cmap=plt.cm.gray)
  plt.show()

In this example, the weights in layer 0 connect the inputs to the first hidden
layer; these weights have one column of 784 values for each hidden node in the
network, so we can iterate over the transpose and put each column -- properly
reshaped -- into a giant image.

That concludes the basic classification example. The ``theanets`` source code
contains a complete ``mnist-classifier.py`` example that you can play around
with.

.. _qs-cli:

Using the Command Line
======================

The ``theanets`` package was designed from the start to use the command line for
configuring most aspects of defining and training a model.

If you work in a command-line environment, you can leave many of the
hyperparameters for your model unspecified when constructing your
:class:`Experiment <theanets.main.Experiment>`, and instead specify the
configuration of your network using flags defined on the command line::

    exp = theanets.Experiment(theanets.Classifier)

This will create the same network as the classification model described above if
you run your file as::

    (venv)~$ mnist-classifier.py --layers 784 100 10

In both cases, the model has one input layer with 784 units, one hidden layer
containing 100 model neurons, and one softmax output layer with 10 units.

.. note::
   Command-line arguments do not work when running ``theanets`` code in IPython;
   within IPython, all parameters must be specified as keyword arguments.

You can set many more hyperparameters on the command line. Use the ``--help``
flag from the command line to show the options that are currently available.

.. _qs-info:

More Information
================

This concludes the quick start guide! Please read more information about
``theanets`` in the :doc:`models` and :doc:`trainers` sections of the
documentation.

The source code for ``theanets`` lives at http://github.com/lmjohns3/theanets.
Please fork, explore, and send pull requests!

Finally, there is also a mailing list for project discussion and announcements.
Subscribe online at https://groups.google.com/forum/#!forum/theanets.
