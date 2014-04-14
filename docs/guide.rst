============
User's Guide
============

This is a user guide!

Basic overview
==============

At its core, ``theanets`` is a tool for optimization. It helps you define,
compute and optimize complex, parametric functions that map a set of input
variables to a set of output variables.

What does this mean for you? Consider an example. Suppose you have some images,
and you'd like to build a classifier model that tells you whether an image
depicts an elephant. In a way, your image data represents measurements of some
set of variables; the fact that there is an elephant (or not) in your images
represents measurements of another variable.

You can use the ``theanets`` package to define a parametric function for
performing this classification task. The parametric function is called the
**model** or the **architecture**. It takes as input a set of measurements (an
image) and yields as output a prediction of whether there is an elephant in the
image. This part is called the **feedforward computation**.

Before you can perform feedforward computations, though, you'll need to find the
parameter settings that best accomplish this task. This is called **training**
or **optimization** and is commonly accomplished using some variation of
[backpropagation]_, which is really a type of of gradient descent.

.. _backpropagation: http://en.wikipedia.org/wiki/Backpropagation

MNIST digits
------------

Before we start, let's first introduce a common dataset that we'll use for
evaluating models.

The `MNIST digits dataset`_ contains 60,000 28x28 images of hand-written digits,
each labeled with its class (0 through 9). This dataset is widely used in the
neural networks community as a standard benchmark. (Even so, the classification
problem represented by this dataset is largely regarded as being solved. The
best models on the MNIST digits dataset currently perform at accuracy levels
exceeding 99.5%.)

.. MNIST digits dataset: http://yann.lecun.com/exdb/mnist/

While it's possible to download the entire MNIST digit dataset and deal with the
binary format, the good folks who work on `Theano`_ have already processed the
dataset into a Python pickle, so we'll use that. You can download the dataset
here:

http://deeplearning.net/data/mnist/mnist.pkl.gz

Save it on your computer as you work along this quick start.

http://www.deeplearning.net/tutorial/gettingstarted.html

Creating an :class:`Experiment`
-------------------------------

The ``theanets`` package contains a helper class, :class:`Experiment`, that
lumps together the definition of a model with the stuff needed to train or
optimize the parameters for that model. The only required argument when you
create an :class:`Experiment` is the constructor for your network. We'll talk
more about networks later; for now, let's assume that you're interested in
detecting elephants in your image data, so you will probably want a
"classifier," or a model that maps your input data onto a variable that
classifies the inputs::

    experiment = theanets.Experiment(theanets.Classifier)
    experiment.run(my_dataset[:1000], my_dataset[1000:])

Creating a :class:`Network`
---------------------------

Command-line Arguments
======================

.. automodule:: theanets.flags


Neural Networks
===============

A feedforward neural network computes a function :math:`f_\\theta: \\mathcal{S}
\\to \\mathcal{T}` between a source space :math:`\\mathcal{S}` and a target
space :math:`\\mathcal{T}` using parameters :math:`\\theta`. To compute
:math:`f`, a network defines a graph of computation units called "artificial
neurons" because they behave, at a very high level, a bit like biological
neurons in animals.
