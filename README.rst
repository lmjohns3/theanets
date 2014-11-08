theanets
========

This package contains implementations of several common neural network
structures, using Theano_ for optimization.

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

There are a few examples in the ``examples/`` directory. Run an example with the
``--help`` flag to get a list of all the command-line arguments; there are many
of them, but some of the notable ones are::

    -n or --layers N1 N2 N3 N4

Build a network with ``N1`` inputs, two hidden layers with ``N2`` and ``N3``
units, and ``N4`` outputs. (Note that this argument is held constant in the
example code, since it needs to correspond to the shape of the data being
processed.)

::
    -g or --hidden-activation logistic|relu|linear|...

Use the given activation function for hidden layer units. (Output layer units
have a linear activation function by default, but an alternative can be given
using the ``--output-activation`` flag.) Several activation functions can be
pipelined together using whitespace or the plus symbol.

::
    -O or --optimize sgd|hf|sgd hf|layerwise hf|...

Use the given optimization method(s) to train network parameters. Several
training methods can be used in sequence by separating their names with spaces
on the command line.

Using the library
-----------------

Probably the easiest way to start with the library is to copy one of the
examples and modify it to perform your tasks. The usual workflow involves
instantiating ``theanets.Experiment`` with a subclass of ``theanets.Network``,
adding some data by calling ``add_dataset(...)``, and finally calling
``train()`` to learn a good set of parameters for your data::

    exp = theanets.Experiment(theanets.Classifier)
    exp.add_dataset('train', my_dataset[:1000])
    exp.add_dataset('valid', my_dataset[1000:])
    exp.train()

You can ``save()`` the trained model to a pickle, or use the trained ``network``
directly to ``predict()`` the outputs on a new dataset::

    print(exp.network.predict(new_dataset))
    exp.save('network-pickle.pkl.gz')

The documentation is relatively sparse, so please file bugs if you find that
there's a particularly hard-to-understand area in the code.
