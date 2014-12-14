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

Package Overview
================

At a high level, the ``theanets`` package is a tool for (a) defining and (b)
optimizing cost functions over a set of data. The workflow in ``theanets``
typically involves two basic steps:

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

The ``theanets`` package provides a helper class, :class:`theanets.Experiment`,
that performs both of these tasks with relatively low effort on your part. You
will typically define a model by creating an experiment with a number of
*hyperparameters* that define the specific behavior of your model. The skeleton
of your code will usually look something like this::

  import matplotlib.pyplot as plt
  import skdata.mnist
  import theanets

  # create an experiment to define and train a model.
  exp = theanets.Experiment(Model,
                            hyperparam1=value1,
                            hyperparam2=value2,
                            ...)

  # train the model.
  exp.train(training_data,
            validation_data,
            optimize='foo',
            ...)

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
implemented in ``theanets``. It's also pretty simple to create custom models
using ``theanets``; see :ref:`hacking-extending` for more information.

.. _qs-classifier:

Classifying MNIST digits
========================

Suppose you're interested in learning a model that can classify an image of an
MNIST digit as a 0, a 1, a 2, etc. For this task, you would normally use the
:ref:`Classifier <models-classification>` feedforward network model. To use this
model in your code, the skeleton above expands like::

  exp = theanets.Experiment(theanets.Classifier,
                            hyperparam1=value1,
                            hyperparam2=value2,
                            ...)

:ref:`Classifier <models-classification>` networks map a layer of
continuous-valued inputs through one or more hidden layers and finally to an
output layer that is activated through the `softmax function`_. The softmax
output is treated as a categorical distribution over the digit labels given the
input image.

.. _softmax function: http://en.wikipedia.org/wiki/Softmax_function

Defining the model
------------------

Next you'll need to set up the hyperparameters for your model. The only required
hyperparameter is ``layers``, which specifies the size of each layer in your
network. The first ("input") and last ("output") layers in your network must
match the size of the data you'll be providing. For the MNIST classification
task, this means your network must have 784 inputs (one for each image pixel)
and 10 outputs (one for each class). For now, we'll focus on models with just
one hidden layer, so for this example you only need to choose a value for the
number of hidden nodes. Let's just choose a nice round number like 100 and see
what happens::

  exp = theanets.Experiment(theanets.Classifier,
                            layers=(784, 100, 10))

There you go, this is all you need to do to define a classifier model that can
be trained up and used. There are many more hyperparameters available, but for
now we'll stick with the defaults.

Learning the parameters
-----------------------

So far, the code above is sufficient to instruct ``theanets`` to create a model.
But models are initialized using small random values for the parameters, which
are unlikely to do anything useful with an MNIST digit as input! To improve the
performance of a model, you'll need to *train* or *optimize* it by adjusting the
model parameters.

The :class:`theanets.Experiment` class handles the general case of training with
fairly little work. Most of the effort required here is in processing your
dataset so that you can use it to train a network.

Preparing a dataset
~~~~~~~~~~~~~~~~~~~

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
1] instead of the default [0, 255]. We've also reshaped the data so that each
example is a 1-dimensional vector rather than a 2-dimensional array of pixels.

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
~~~~~~~~~~~~~~~~~~~~~

The next step is to specify the training algorithm to use, and any associated
hyperparameter values. This is most naturally accomplished using the
``train`` method of the experiment object::

    exp.train(training_data,
              optimize='nag',
              learning_rate=1e-3,
              momentum=0.9)

The first positional argument to this method is the training dataset, and the
second (if provided) is a validation dataset. (These positional arguments can
also be passed to the ``train()`` method using the keywords ``train_set`` and
``valid_set``, respectively.) If a validation dataset is not provided, the
training dataset will be used for validation.

The ``optimize`` keyword argument specifies an algorithm to use for training.
(If you do not provide a value for this argument, ``'rmsprop'`` is used by
default.) Any subsequent keyword arguments will be passed to the training
algorithm implementation; these arguments typically specify hyperparameters of
the training algorithm like the learning rate and so forth.

The available training methods are described in :doc:`trainers`; here we've used
Nesterov's Accelerated Gradient [Sut13]_, a type of stochastic gradient descent
with momentum.

To train our model, we will use the MNIST digits dataset from above. Putting
everything together yields code that looks like this::

  train, valid, _ = load_mnist()
  exp = theanets.Experiment(theanets.Classifier, layers=(784, 100, 10))
  exp.train(train, valid, optimize='nag', learning_rate=1e-3, momentum=0.9)

If you put this code, plus any necessary imports, into a file called something
like ``mnist-classifier.py``, and then run it on the command-line, your computer
will do a bunch of work to learn good parameter values for your model! If you
enable Python's ``logging`` module you'll also get updates on the console about
the progress of the optimization procedure.

Displaying learned features
---------------------------

Once you've trained a model, you will probably want to do something useful with
it. For classifiers, you can obtain predictions on new data::

  exp.network.classify(new_dataset)

You can also create a plot of the features that the model learns::

  img = np.zeros((28 * 10, 28 * 10), dtype='f')
  for i, pix in enumerate(exp.network.get_weights(0).T):
      r, c = divmod(i, 10)
      img[r * 28:(r+1) * 28, c * 28:(c+1) * 28] = pix.reshape((28, 28))
  plt.imshow(img, cmap=plt.cm.gray)
  plt.show()

After the model has been trained, the weights connecting the input to the hidden
layer are available using ``exp.network.get_weights(...)``. The weights in layer
0 connect the inputs to the first hidden layer; in this example these weights
have one column of 784 values for each hidden node in the network, so we can
iterate over the transpose and put each column -- properly reshaped into a 28×28
pixel array -- into a giant image.

That concludes the basic classification example. The ``theanets`` source code
contains a complete ``mnist-classifier.py`` example that you can play around
with.

.. _qs-autoencoder:

Encoding MNIST digits
=====================

Some types of neural network models display a powerful ability to learn useful
features from a set of data without requiring any label information. Often
referred to as feature learning or manifold learning, this ability is useful
because labeled data (e.g., images annotated with the objects in them) are often
difficult to obtain, while unlabeled data (e.g., images) are relatively easy to
find.

A class of neural network architectures known as autoencoders can perform such a
learning task; an autoencoder takes as input a data sample and attempts to
produce the same data sample as its output. Mathematically, an autoencoder with
a single hidden layer can be expressed using this forward transform:

.. math::
   f(x) = g_o(W_o g_h(W_h x + b_h) + b_o)

Here, :math:`g_i`, :math:`W_i`, and :math:`b_i` are the activation function,
weights, and bias of layer :math:`i` in the network. The trainable parameters
are :math:`\theta = (W_o, W_h, b_o, b_h)`.

To train the weights and biases in the network, an autoencoder typically
optimizes a squared-error reconstruction loss:

.. math::
   \ell(x) = \left\| f(x) - x \right\|_2^2 + \lambda R(\theta, x)

Where :math:`R()` is some regularizer that helps prevent the model from
overfitting.

This optimization process could result in a trivial model, depending on the
setup of the network; for example, with linear activations :math:`g_o(z) =
g_h(z) = z`, identity weights :math:`W_o = W_h = I`, and zero bias :math:`b_o =
b_h = 0`, an autoencoder implements the identity transform:

.. math::
   f(x) = x

Similarly, even if the hidden unit activations are nonlinear, the network is
capable of learning an identity transform. But things get much more interesting
when the network is forced to reproduce the input under some constraint.

One popular form of constraint is dimensionality reduction, which forces the
network to project its input into a lower-dimensional space and then project it
back to the original dimensionality. With linear hidden activations, tied
weights, and no bias, this model will recover the same subspace as PCA:

.. math::
   \ell = \left\| WW^\top x - x \right\|_2^2

After all, PCA is by definition the subspace that preserves the most variance in
the data! This model limits us to at most :math:`d` features, however (where the
elements of :math:`x` are :math:`d`-dimensional). Let's see what else is
possible.

If instead we wanted to learn an overcomplete feature set (i.e., more than
:math:`d` features), we could encourage the model to learn a non-trivial
representation of the data by adding a regularizer that specifies how the
features should behave. For instance, if we require that the model reproduce the
input data using as little feature representation as possible, we could add an
:math:`\ell_1` penalty to the hidden representation:

.. math::
   \ell = \left\| WW^\top x - x \right\|_2^2 + \lambda \left\| W^\top x \right\|_1

Le et al. showed that this model is actually equivalent to ICA.

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

You can set many more hyperparameters on the command line. Use the ``--help``
flag from the command line to show the options that are currently available.

More information
================

This concludes the quick start guide! Please read more information about
``theanets`` in the :doc:`models` and :doc:`trainers` sections of the
documentation.

References
==========

.. [Sut13] I Sutskever, J Martens, G Dahl, GE Hinton. "On the importance of
           initialization and momentum in deep learning." *Proc ICML*, 2013.
           http://jmlr.csail.mit.edu/proceedings/papers/v28/sutskever13.pdf
