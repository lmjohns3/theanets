# theano-nets

This package contains implementations of several common neural network
structures, using the amazing [Theano][] package for optimization.

[Theano]: http://deeplearning.net/software/theano/

## Installation

Install the latest published code using pip:

    pip install lmj.nn

Or download the current source and run it from there:

    git clone http://github.com/lmjohns3/theano-nets
    cd theano-nets
    python setup.py develop

## Getting started

There are a few examples in the `examples/` directory. Run an example with the
`--help` flag to get a list of all the command-line options ; there are many of
them, but some of the notable ones are :

    -n or --layers N1 N2 N3 N4

Build a network with `N1` inputs, two hidden layers with `N2` and `N3` units,
and `N4` outputs.

    -g or --activation logistic|relu|linear|norm:mean+logistic|...

Use the given activation function for hidden layer units. (All output layer
units have a linear activation function.) Several activation functions can be
pipelined together using `+`.

    -O or --optimize sgd|hf|sgd+hf|...

Use the given optimization method to train network parameters. Several training
methods can be used in sequence by concatenating their names with `+`.

## Using the library

The easiest way to start with the library is to copy one of the examples and
modify it to perform your tasks. The usual workflow involves subclassing
`lmj.nn.Main` and providing implementations of `get_network(self)` (which should
return a Python class derived from `lmj.nn.Network`) and `get_datasets(self)`
(which should return a (training, validation) tuple of data). You'll need to
make sure the dimensions of everything match up properly.
