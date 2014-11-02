==========
Quickstart
==========

This page provides a quick overview of the ``theanets`` package that should get
you up and running with a few simple examples. Once you understand the basic
workflow, you will be able to extend the examples to your own datasets and
modeling problems. Along the way, there are links to the reference documentation
and to the `User's Guide <http://theanets.readthedocs.org/en/latest/guide.html>`_,
which expands on the ideas presented here.

If you find an error in these documents, or just want to clarify something,
please file an issue or send a pull request to
https://github.com/lmjohns3/theano-nets.

.. _qs-setup:

Getting started
===============

First of all, you'll probably want to download and install ``theanets``. You can
do this any number of ways, but the easiest is to use ``pip``::

    pip install theanets

This command will automatically install all of the dependencies for
``theanets``, including ``numpy`` and ``Theano``.

MNIST digits
------------

.. image:: http://www.heikohoffmann.de/htmlthesis/img679.gif

The examples on this page will all use the `MNIST digits dataset
<http://yann.lecun.com/exdb/mnist/>`_. If you want to follow along, you can
download a ready-to-go Python pickle of the dataset from
http://deeplearning.net/data/mnist/mnist.pkl.gz.

.. _qs-quickstart:

Creating a model
================

The ``theanets`` package is a tool for (a) defining and (b) optimizing cost
functions over a set of data. As such, the workflow in ``theanets`` involves two
basic steps:

#. First, you need to define the structure of the model that you'll use for your
   task. For instance, if you're trying to classify MNIST digits, then you'll
   want something that takes in pixels and outputs digit classes (a
   "classifier"). If you're trying to model the digit images without labels, you
   might want to use a model that takes in pixels and outputs pixels (an
   "autoencoder").
#. Second, you will likely need to train or adjust the parameters in your model
   so that it has a low cost or performs well with respect to some benchmark.
   For classification, you might want to adjust your model parameters to
   minimize the negative log-likelihood of the correct image class, and for
   autoencoders you might want to minimize the reconstruction error.

The ``theanets`` package provides a helper class, :class:`theanets.Experiment`,
that is designed to perform both of these tasks with relatively low effort on
your part. We'll look at these steps below.

Classifier
----------

To classify a dataset like the MNIST digits, you can use the
:class:`theanets.Classifier` feedforward network. These networks map a layer of
continuous-valued inputs through a series of hidden layers and finally to a
softmax output layer.

The MNIST digits each contain 784 values (28 by 28 pixels = 784 variables), and
each digit falls into one of 10 classes. These values determine the number of
input and output units in our network. The hidden structure---the number and
size of layers between the input and output---can be determined by other
constraints of the problem.

To create an appropriate classification network, use the ``layers`` keyword
argument when creating an :class:`theanets.Experiment` in ``theanets``::

    experiment = theanets.Experiment(
        theanets.Classifier,
        layers=(784, 100, 10),
    )

In this example, the network will have one input layer containing 784 neurons,
one hidden layer containing 100 neurons, and one output layer containing 10
neurons. You can give your network three hidden layers simply by adjusting the
value of the ``layers`` argument, e.g. ``layers=(784, 500, 200, 100, 10)``.

Command-line arguments
^^^^^^^^^^^^^^^^^^^^^^

If you work in a command-line environment, you can leave the ``layers``
unspecified when constructing your experiment, and instead the value of the
``--layers`` command-line flag will be used::

    experiment = theanets.Experiment(theanets.Classifier)

This will create the same network as above if you run your file as::

    my-classifier.py --layers 784 100 10

In both cases, the model has one hidden layer containing 100 model neurons.

You can add as many numbers as desired to the ``layers`` sequence, but keep in
mind that more layers yields models that:

- consume more resources (both memory and processing time),
- are more prone to overfitting, and
- are more difficult to train.

Autoencoder
-----------

An autoencoder is a machine learning model that attempts to reproduce its input
as its output. You can create such a model by using the
:class:`theanets.Autoencoder` class::

    experiment = theanets.Experiment(
        theanets.Autoencoder,
        layers=(784, 100, 784),
    )

Here, the number of layers is the same as in the classifier example (one hidden
layer), but the number of output units has changed, because the autoencoder will
attempt to reproduce its input.

Regressor
---------

The third major class of model in ``theanets`` is the
:class:`theanets.Regressor`. This type of model is like the
:class:`theanets.Classifier`, but instead of attempting to produce a one-of-k
output using the softmax, a :class:`theanets.Regressor` attempts to output some
continuous-valued target vector for each input.

This type of model isn't usually useful for the MNIST digits dataset, so we
won't talk about it further here. Keep in mind that it exists, however, for
those moments when you need a powerful nonlinear regression model.

Custom models
-------------

It's also pretty simple to create custom models using ``theanets``. Please see
more information in the `User's Guide
<http://theanets.readthedocs.org/en/latest/guide.html>`_.

Training models
===============

So far, you've seen how to create models. But models are created using small
random values for the parameters, which are unlikely to do anything useful with
an MNIST digit! To improve the performance of a model, you'll need to **train**
it by adjusting the model parameters so that the error of the model output
decreases.

The :class:`theanets.Experiment` class handles the general case of training with
fairly little work. Most of the effort required here is in processing your
dataset so that you can use it to train a network.

More information
================

This concludes the quick start guide! Please read more information about
``theanets`` in the `User's Guide <http://theanets.readthedocs.org/en/latest/guide.html>`_.

.. _qs-cli:

Using the Command Line
======================
