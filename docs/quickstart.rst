===============
Getting Started
===============

This page provides a quick overview of the ``theanets`` package. It is aimed at
getting you up and running with a few simple examples. Once you understand the
basic workflow, you will be able to extend the examples to your own datasets and
modeling problems.

This document does not explain the theory behind most of the models that are
implemented in the ``theanets`` package. Instead, it contains links links to the
reference documentation, which expands on the ideas presented here.

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
``theanets``, including ``numpy`` and ``Theano``.

MNIST digits
------------

The examples throughout the documentation use the `MNIST digits dataset
<http://yann.lecun.com/exdb/mnist/>`_, a set of 70,000 28×28 images of
hand-written digits. Each MNIST digit is labeled with the correct digit class
(0, 1, ... 9).

.. image:: http://www.heikohoffmann.de/htmlthesis/img679.gif

Because the MNIST digits are high-dimensional pixel arrays, they are useful for
testing out models of unsupervised learning like autoencoders. But because the
MNIST digits are also labeled, they are useful for testing out models of
supervised learning like classifiers. We'll address both of these tasks as
examples in this document.

You can download a copy of the dataset by using the ``skdata`` package::

    pip install skdata
    pip install matplotlib

To show a few of the digits, for example::

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

    # some imports -- we will leave these out of
    # subsequent code blocks.
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
    exp.add_trainer(...)
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

It's also pretty simple to create custom models using ``theanets``, but this is
not needed to get started. Please see :ref:`hacking-extending` for more
information about extending the existing models. :doc:`models` contains detailed
documentation about each of the types of models implemented in ``theanets``.

.. _one-hot: http://en.wikipedia.org/wiki/One-hot

.. _qs-classifier:

Classifying MNIST digits
========================

Suppose you're interested in learning a model that can classify an image of an
MNIST digit as a 0, a 1, a 2, etc. For this task, you can use the
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
focus on models with just one hidden layer, so you need to choose a value for
the number of hidden nodes. Let's just choose a nice round number like 100 and
see what happens::

    exp = theanets.Experiment(
        theanets.Classifier,
        layers=(784, 100, 10),
    )

This is already close to a model that can be trained up and used. In this
example, the classifier network will have one input layer containing 784
neurons, one hidden layer containing 100 neurons, and one softmax output layer
containing 10 neurons.

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



.. _qs-autoencoder:

Learning an autoencoder
=======================

.. _qs-deepautoencoder:

Learning a deep autoencoder
===========================

You can give your network three hidden layers simply by adjusting the value of
the ``layers`` argument, e.g. ``layers=(784, 500, 200, 100, 10)``. You can add
as many numbers as desired to the ``layers`` sequence, but keep in mind that
including more layers often tends to yield models that:

- consume more resources (both memory and processing time), and
- are more difficult to train.

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

    (venv)~$ my-classifier.py --layers 784 100 10

In both cases, the model has one input layer with 784 units, one hidden layer
containing 100 model neurons, and one softmax output layer with 10 units.

More information
================

This concludes the quick start guide! Please read more information about
``theanets`` in the :doc:`models` and :doc:`trainers` sections of the
documentation.
