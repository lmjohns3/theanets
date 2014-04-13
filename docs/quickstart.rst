==========
Quickstart
==========

This page provides a quick overview of the ``theanets`` package that should get
you up and running with a few simple examples. Once you understand the basic
workflow, you will be able to extend the examples to your own datasets and
modeling problems!

Along the way, there are links to the reference documentation and to the `User's
Guide`_, which expands on the ideas here to hopefully make them more accessible.

.. User's Guide: http://theanets.readthedocs.org/en/latest/guide.html

If you find an error in these documents, or just want to clarify something,
please file an issue or send a pull request to
https://github.com/lmjohns3/theano-nets and we'll work on fixing it up!

:mod:`theanets`

MNIST digits
============

These examples will use the `MNIST digits dataset`_. If you want to follow
along, you can download a Python pickle of the dataset from:

http://deeplearning.net/data/mnist/mnist.pkl.gz

.. MNIST digits dataset: http://yann.lecun.com/exdb/mnist/

Creating models
---------------

Generally, it's easiest to use `theanets` to create an :class:`Experiment`, and
then use the :class:`Experiment` to define and train a model.

Classifier
==========

To classify a dataset like the MNIST digits, you can use the :class:`Classifier`
feedforward network. These networks map a layer of continuous-valued inputs
through a series of hidden layers and finally to a softmax output layer.

The MNIST digits each contain 784 values (28 pixels by 28 pixels = 784
variables), and each digit falls into one of 10 classes. These values determine
the number of input and output units in our network. The hidden structure---the
number and size of layers between the input and output---can be determined by
other constraints of the problem.

To create an appropriate classification network with just one hidden layer, use
the ``layers`` keyword argument when setting your experiment::

    experiment = theanets.Experiment(
        theanets.Classifier,
        layers=(784, 100, 10),
    )

Command-line arguments
~~~~~~~~~~~~~~~~~~~~~~

Alternatively, if you work in a command-line environment, you can leave the
``layers`` unspecified when constructing your experiment, and instead the value
of the ``--layers`` command-line flag will be used::

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
as its output. You can create such a model by using the :class:`Autoencoder`
class::

    experiment = theanets.Experiment(
        theanets.Autoencoder,
        layers=(784, 100, 784),
    )

Here, the number of layers is the same as in the classifier example (one hidden
layer), but the number of output units has changed, because the autoencoder will
attempt to reproduce its input.

Regressor
---------

The third major class of model in `theanets` is the :class:`Regressor` class.
This type of model is like the :class:`Classifier`, but instead of attempting to
produce a one-of-k output using the softmax, a :class:`Regressor` attempts to
output some continuous-valued target vector for each input.

This type of model isn't usually useful for the MNIST digits dataset, so we
won't talk about it further here. Keep in mind that it exists, however, for
those moments when you need a powerful nonlinear regression model.

Custom models
-------------

It's also pretty simple to create custom models using `theanets`, but the quick
start is not where to learn about it. Please see more information in the `User's
Guide`_.

Training models
===============

So far, you've seen how to create models. But models are created using small
random values for the parameters, which are unlikely to do anything useful with
an MNIST digit! To improve the performance of a model, you'll need to **train**
it by adjusting the model parameters so that the error of the model output
decreases.

The :class:`Experiment` class handles the general case of training with fairly
little work. Most of the effort required here is in processing your dataset so
that you can use it to train a network.

More information
================

This concludes the quick start guide! Please read more information about
`theanets` in the `User's Guide`_.
