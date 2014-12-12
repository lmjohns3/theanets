theanets
========

This package contains implementations of several common neural network
structures, using Theano_ for optimization, symbolic differentiation, and
transparent GPU computations. Some things it does:

- Provides several common neural network models:
  - Feedforward Classifier, Autoencoder, Regression
  - Recurrent Classifier, Autoencoder, Regression, Prediction
  - Easy to specify models with any number of layers
- Allows for many different types of regularization:
  - L1 and L2 weight decay
  - L1 and L2 hidden activation penalties (e.g., sparse autoencoders)
  - Dropout on inputs (e.g., denoising autoencoders)
  - Dropout on hidden units
  - Noise on inputs (e.g., denoising autoencoders)
  - Noise on hidden units
  - Implement custom regularization with a bit of Python code
- Implements several optimization algorithms:
  - SGD variants: NAG, Rprop, RmsProp
  - Many algorithms in ``scipy.optimize.minimize``
  - Hessian-Free (not currently compatible with Python3)
  - Greedy layerwise pre-training
- Compatible with Python2 and Python3

And here are some things it does not do:

- RBMs
- Convolutions
- Maxout

.. _Theano: http://deeplearning.net/software/theano/

Installation
------------

Install the latest published code using pip::

    pip install theanets

Or download the current source and run it from there::

    git clone http://github.com/lmjohns3/theano-nets
    cd theano-nets
    python setup.py develop

Getting started
---------------

There are a few example scripts in the ``examples`` directory. You can run these
from the command-line::

    python examples/mnist-autoencoder.py

This example trains an autoencoder with a single hidden layer to reconstruct
images of handwritten digits from the MNIST dataset.

Command-line configuration
--------------------------

The ``theanets`` package comes built-in with several network models and
optimization algorithms available. Many of the available options can be
configured from the command-line. To get help on the command-line options, run
an example with the ``--help`` flag::

    python examples/mnist-autoencoder.py

There are many arguments, but some of the notable ones are::

    -n or --layers N1 N2 N3 N4

Builds a network with ``N1`` inputs, two hidden layers with ``N2`` and ``N3``
units, and ``N4`` outputs. (Note that this argument is held constant in most of
the examples, since it needs to correspond to the shape of the data being
processed.)

::
    -g or --hidden-activation logistic|relu|linear|...

Use the given activation function for hidden layer units. (Output layer units
have a linear activation function by default, but an alternative can be given
using the ``--output-activation`` flag.) Several activation functions can be
pipelined together using the plus symbol.

::
    -O or --optimize sgd|hf|sgd hf|layerwise hf|...

Use the given optimization algorithm(s) to train network parameters. Several
training algorithms can be used in sequence by separating their names with
spaces on the command line.

Using the library
-----------------

Probably the easiest way to start with the library is to copy one of the
examples and modify it to perform your tasks. The usual workflow involves
instantiating ``theanets.Experiment`` with a subclass of ``theanets.Network``,
and then calling ``train()`` to learn a good set of parameters for your data::

    exp = theanets.Experiment(theanets.Classifier)
    exp.train(my_dataset[:1000], my_dataset[1000:])

You can ``save()`` the trained model to a pickle, or use the trained ``network``
directly to ``predict()`` the outputs on a new dataset::

    print(exp.network.predict(new_dataset))
    exp.save('network-pickle.pkl.gz')

Documentation and support
-------------------------

The `package documentation`_ lives at readthedocs. The documentation is
relatively sparse, so please file bugs if you find that there's a particularly
hard-to-understand area in the code.

For project announcements and discussion, subscribe to the
`project mailing list`_.

.. _package documentation: http://theanets.readthedocs.org
.. _project mailing list: https://groups.google.com/forum/#!forum/theanets

