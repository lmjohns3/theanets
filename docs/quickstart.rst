===============
Getting Started
===============

This page provides a quick overview of the ``theanets`` package. It is aimed at
getting you up and running with a few simple examples. Once you understand the
basic workflow, you will be able to extend the examples to your own datasets and
modeling problems.

This document does not explain the theory behind most of the models or
optimization algorithms that are implemented in the ``theanets`` package.
Instead, it contains links links to the reference documentation, which expands
on the ideas presented here.

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

Overview
========

At a high level, the ``theanets`` package is a tool for (a) defining and (b)
optimizing cost functions over a set of data. The workflow in ``theanets``
typically involves two basic steps:

#. First, you define the structure of the model that you'll use for your task.
   For instance, if you're trying to classify MNIST digits, then you'll want
   something that takes in pixels and outputs digit classes (a "classifier"). If
   you're trying to model the digit images without labels, you might want to use
   a model that takes in pixels and outputs pixels (an "autoencoder").
#. Second, you train or adjust the parameters in your model so that it has a low
   cost or performs well with respect to some benchmark. For classification, you
   might want to adjust your model parameters to minimize the negative
   log-likelihood of the correct image class, and for autoencoders you might
   want to minimize the reconstruction error.

The ``theanets`` package provides a helper class, :class:`theanets.Experiment`,
that is designed to perform both of these tasks with relatively low effort on
your part. You will typically define a model by creating an experiment with a
number of *hyperparameters* that define the specific behavior of your model. The
skeleton of your code will usually look something like this::

  # some imports -- we will omit these
  # from subsequent code blocks.
  import matplotlib.pyplot as plt
  import skdata.mnist
  import theanets

  # create an experiment to define and train a model.
  exp = theanets.Experiment(
      Model,
      hyperparam1=value1,
      hyperparam2=value2,
      ...)

  # add an optimization algorithm to learn model parameters.
  exp.add_trainer(...)

  # train the model.
  exp.run(training_data, validation_data)

  # use the trained model.
  model = exp.network
  model.predict(test_data)

Several broad classes of models are pre-defined in ``theanets``:

- :ref:`Classifier <models-classification>`: A model that maps its input onto a
  (usually small) set of output nodes that represent the probability of a label
  given the input.
- :ref:`Autoencoder <models-autoencoders>`: A model that attempts to reproduce
  its input as its output.
- :ref:`Regressor <models-regression>`: Like the classifier, but instead of
  attempting to produce a `one-hot`_ output, a regressor attempts to produce
  some continuous-valued target vector for each input.

.. _one-hot: http://en.wikipedia.org/wiki/One-hot

:doc:`models` contains detailed documentation about each of the types of models
implemented in ``theanets``.

It's also pretty simple to create custom models using ``theanets``, but this is
not needed to get started. Please see :ref:`hacking-extending` for more
information about extending the existing models.

.. _qs-classifier:

Classifying MNIST digits
========================

Suppose you're interested in learning a model that can classify an image of an
MNIST digit as a 0, a 1, a 2, etc. For this task, you would normally use the
:ref:`Classifier <models-classification>` feedforward network model. To use this
model in your code, the skeleton above expands like::

  exp = theanets.Experiment(
      theanets.Classifier,  # use the classifier model type.
      hyperparam1=value1,
      hyperparam2=value2,
      ...)

Defining the classifier
-----------------------

Now what about the hyperparameters? :ref:`Classifier <models-classification>`
networks map a layer of continuous-valued inputs through one or more hidden
layers and finally to an output layer that is activated through the `softmax
function`_. The softmax output is treated as a categorical distribution over the
digit labels given the input image.

.. _softmax function: http://en.wikipedia.org/wiki/Softmax_function

So the first hyperparameter that you'll need to set is ``layers``, which
specifies the number and size of each layer in your network. For this example,
the size of the MNIST images (784) determines the number of input nodes you
need, and the number of digit classes (10) determines the output. For now, we'll
focus on models with just one hidden layer, so you only need to choose a value
for the number of hidden nodes. Let's just choose a nice round number like 100
and see what happens::

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(784, 100, 10))

This is all you need to do to define a classifier model that can be trained up
and used. There are many more hyperparameters available, but for now we'll stick
with the defaults for most of them.

Training the classifier
-----------------------

So far, the code above is sufficient to instruct ``theanets`` to create a model.
But models are created using small random values for the parameters, which are
unlikely to do anything useful with an MNIST digit as input! To improve the
performance of a model, you'll need to *train* it by adjusting the model
parameters so that the error of the model output decreases.

The :class:`theanets.Experiment` class handles the general case of training with
fairly little work. Most of the effort required here is in processing your
dataset so that you can use it to train a network.

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

Here we've rescaled the image data so that each pixel lies in the interval [0,
1] instead of the default [0, 255]. The load function returns a training split
(the first 50000 images), a validation split (the remainder of the training data
from ``skdata``), and a test split (the test split from ``skdata``).

.. note::

   Because ``theanets`` uses Theano for its computations, most datasets need to
   be cast to a value that is compatible with your setting for
   `Theano's "floatX" configuration parameter`_. Unless you have a really
   expensive GPU, this is likely to mean that you need to use 32-bit floats.

.. _Theano's "floatX" configuration parameter: http://deeplearning.net/software/theano/library/config.html#config.floatX

The next step is to specify the training algorithm to use, and any associated
hyperparameter values. This is most naturally accomplished using the
``add_trainer`` method of the experiment object::

    exp.add_trainer('nag', learning_rate=1e-3, momentum=0.9)

The first argument to the method is the name of a training algorithm, and any
subsequent keyword arguments will be passed to the training code. The available
training methods are described in :doc:`trainers`; here we've used Nesterov's
Accelerated Gradient, a type of stochastic gradient descent with momentum.

Finally, the model needs to be trained before it can be used. Putting everything
together yields code that looks like this::

  train, valid, _ = load_mnist()
  exp = theanets.Experiment(theanets.Classifier, layers=(784, 100, 10))
  exp.add_trainer('nag', learning_rate=1e-3, momentum=0.9)
  exp.run(train, valid)

If you put this code (plus any necessary imports) into a file called, say,
``mnist-classifier.py``, and then run it on the command-line, your computer will
do a bunch of work to learn good parameter values for your model ... and then it
will throw it all away!

Displaying learned features
---------------------------

Let's get this example to do something useful by showing a plot of the
"features" that the model learns::

  img = np.zeros((28 * 10, 28 * 10), dtype='f')
  for i, pix in enumerate(exp.network.weights[0].get_value().T):
      r, c = divmod(i, 10)
      img[r * 28:(r+1) * 28, c * 28:(c+1) * 28] = pix.reshape((28, 28))
  plt.imshow(img, cmap=plt.cm.gray)
  plt.show()

After the model is trained, we've accessed the weights connecting the input to
the hidden layer using ``exp.network.weights[0]``. This value is a Theano shared
array, so to get its current value we need to call ``.get_value()``. This array
has one column of 784 values for each hidden node in the network, so we can
iterate over the transpose and put each column -- properly reshaped into a 28×28
pixel array -- into a giant image and then just plot that image.

The ``theanets`` source code contains a complete ``mnist-classifier.py`` example
that you can play around with. In addition, there are also examples of using
:class:`theanets.Autoencoder` and "deep" autoencoders for learning features from
the MNIST digits.

.. _qs-cli:

Using the command line
======================

The ``theanets`` package was designed from the start to use the command line for
configuring most aspects of defining and training a model.

If you work in a command-line environment, you can leave many of the
hyperparameters for your model ``layers`` unspecified when constructing your
:class:`theanets.Experiment`, and instead specify the configuration of your
network using flags defined on the command line::

    exp = theanets.Experiment(theanets.Classifier)

This will create the same network as the classification model above if you run
your file as::

    (venv)~$ mnist-classifier.py --layers 784 100 10

In both cases, the model has one input layer with 784 units, one hidden layer
containing 100 model neurons, and one softmax output layer with 10 units.

More information
================

This concludes the quick start guide! Please read more information about
``theanets`` in the :doc:`models` and :doc:`trainers` sections of the
documentation.
