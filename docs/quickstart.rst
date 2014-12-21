===============
Getting Started
===============

This page provides a quick overview of the ``theanets`` package. It is aimed at
getting you up and running with a few simple examples. Once you understand the
basic workflow, you will be able to extend the examples to your own datasets and
modeling problems.

This document does not explain the theory behind most of the models or
optimization algorithms that are implemented in the ``theanets`` package.
Instead, it contains links to the reference documentation, which expands on the
ideas presented here.

If you find an error in any of the documentation, or just want to clarify
something, please file an issue or send a pull request to
https://github.com/lmjohns3/theano-nets.

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

.. _qs-overview:

Package overview
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
with a number of *hyperparameters* that define the specific behavior of your
model. The skeleton of your code will usually look something like this::

  import matplotlib.pyplot as plt
  import skdata.mnist
  import theanets

  # create an experiment to define and train a model.
  exp = theanets.Experiment(
      Model,
      hyperparam1=value1,
      hyperparam2=value2,
      ...)

  # train the model.
  exp.train(
      training_data,
      validation_data,
      optimize='foo',
      ...)

  # use the trained model.
  model = exp.network
  model.predict(test_data)

This quickstart document guides you through the three main stages below.

.. _qs-creating:

================
Creating a Model
================

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

:doc:`models` contains detailed documentation about each of the types of models
implemented in ``theanets``, including mathematical background to help
understand what each model tries to do.

It's also pretty simple to create custom models using ``theanets``; see
:ref:`hacking-extending` for more information.

.. _qs-classifier:

Classifiers
===========

Suppose you're interested in learning a model that can classify an image of an
MNIST digit as a 0, a 1, a 2, etc. For this task, you would normally use the
:class:`theanets.feedforward.Classifier` feedforward network model.

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

.. _qs-autoencoder:

Autoencoders
============

The ``theanets`` package also provides an
:class:`theanets.feedforward.Autoencoder` class to construct models that can
learn features from data without labels. An autoencoder for MNIST digits, for
example, takes as input an unlabeled MNIST digit image and then attempts to
produce this same digit image as output. The hidden layers in such a model are
then called the "features" of the data that the model learns.

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

Finally, some autoencoders are capable of using *tied weights*, which means the
"input" weights are the same as the "output" weights in the model. Autoencoders
with tied weights represent some very common machine learning algorithms; see
:ref:`models-tied-weights` for more information.

Defining the model
==================

Having chosen a model class to use for your task, and a set of layer sizes that
you want in your model, you will create a :class:`theanets.main.Experiment` to
construct your model.

There are two required arguments: the class of the model to create, and the
``layers`` keyword argument, which specifies the number and size of the layers
in your network.  define a classifier model::

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(784, 100, 10))

This is all you need to do to define a classifier model that can be trained up
and used. There are many more hyperparameters available, but for now we'll stick
with the defaults.

If you want to set up a more sophisticated model like a denoising autoencoder,
you can add regularization hyperparameters when you create your experiment::

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(784, 1000, 784),
      input_noise=0.1)

Here we've specified that our model has a single, overcomplete hidden layer, and
gaussian noise with standard deviation 0.1 will be added the the inputs. To
create a sparse autoencoder, just replace the ``input_noise`` keyword argument
with ``hidden_l1``, which specifies the amount of penalty that should be applied
to the hidden unit activation.

.. _qs-training:

================
Training a Model
================

So far, the code above is sufficient to instruct ``theanets`` to create a model.
But models are initialized using small random values for the parameters, which
are unlikely to do anything useful with an MNIST digit as input! To improve the
performance of a model, you'll need to *train* or *optimize* it by adjusting the
model parameters.

The :class:`theanets.main.Experiment` class handles the general case of training
with fairly little work. Most of the effort required here is in processing your
dataset so that you can use it to train a network.

Preparing a dataset
===================

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
=====================

The next step is to specify the training algorithm to use, and any associated
hyperparameter values. This is most naturally accomplished using the
``train`` method of the experiment object::

  exp.train(training_data,
            optimize='nag',
            learning_rate=1e-3,
            momentum=0.9)

The first positional argument to this method is the training dataset, and the
second (if provided) is a validation dataset. (These positional arguments can
also be passed to the :func:`theanets.main.Experiment.train` method using the
keywords ``train_set`` and ``valid_set``, respectively.) If a validation dataset
is not provided, the training dataset will be used for validation.

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
---------------------

The :func:`theanets.main.Experiment.train` method is actually just a thin
wrapper over the underlying :func:`theanets.main.Experiment.itertrain` method,
which you can use directly if you want to do something special during training::

  for costs in exp.itertrain(train, valid, **kwargs):
      print(costs['J'])

Trainers yield a dictionary after each training iteration. The keys and values
in each dictionary give the costs and monitors that are computed during
training, which will vary depending on the model and the training algorithm.
However, there will always be a ``'J'`` key that gives the value of the loss
function that is being optimized. For classifier models, the dictionary will
also have an ``'acc'`` key, which gives the percent accuracy of the classifier
model.

Saving and loading
==================

The :class:`theanets.main.Experiment` class can snapshot your model
automatically during training. When you call
:func:`theanets.main.Experiment.train`, you can provide the following keyword
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

You can also save and load models manually by calling
:func:`theanets.main.Experiment.save` and :func:`theanets.main.Experiment.load`,
respectively.

.. _qs-using:

=============
Using a Model
=============

Once you've trained a model, you will probably want to do something useful with
it. For classifiers, you can obtain predictions on new data using the
:func:`theanets.feedforward.Classifier.classify` method::

  exp.network.classify(new_dataset)

You can also create a plot of the features that the model learns::

  img = np.zeros((28 * 10, 28 * 10), dtype='f')
  for i, pix in enumerate(exp.network.get_weights(0).T):
      r, c = divmod(i, 10)
      img[r * 28:(r+1) * 28, c * 28:(c+1) * 28] = pix.reshape((28, 28))
  plt.imshow(img, cmap=plt.cm.gray)
  plt.show()

After the model has been trained, the weights connecting the input to the hidden
layer are available using :func:`theanets.feedforward.Network.get_weights`. The
weights in layer 0 connect the inputs to the first hidden layer; in this example
these weights have one column of 784 values for each hidden node in the network,
so we can iterate over the transpose and put each column -- properly reshaped
into a 28×28 pixel array -- into a giant image.

That concludes the basic classification example. The ``theanets`` source code
contains a complete ``mnist-classifier.py`` example that you can play around
with.

.. _qs-misc:

=============
Miscellaneous
=============

Using the Command Line
======================

The ``theanets`` package was designed from the start to use the command line for
configuring most aspects of defining and training a model.

If you work in a command-line environment, you can leave many of the
hyperparameters for your model unspecified when constructing your
:class:`theanets.main.Experiment`, and instead specify the configuration of your
network using flags defined on the command line::

    exp = theanets.Experiment(theanets.Classifier)

This will create the same network as the classification model described above if
you run your file as::

    (venv)~$ mnist-classifier.py --layers 784 100 10

In both cases, the model has one input layer with 784 units, one hidden layer
containing 100 model neurons, and one softmax output layer with 10 units.

You can set many more hyperparameters on the command line. Use the ``--help``
flag from the command line to show the options that are currently available.

More Information
================

This concludes the quick start guide! Please read more information about
``theanets`` in the :doc:`models` and :doc:`trainers` sections of the
documentation.

The source code for ``theanets`` lives at http://github.com/lmjohns3/theanets.
Please fork, explore, and send pull requests!

Finally, there is also a mailing list for project discussion and announcements.
Subscribe online at https://groups.google.com/forum/#!forum/theanets.
