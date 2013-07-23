# theano-nets

This package contains implementations of several common neural network
structures, using the amazing [Theano][] package for optimization.

[Theano]: http://deeplearning.net/software/theano/

## Installation

Install the latest published code using pip:

    pip install theanets

Or download the current source and run it from there:

    git clone http://github.com/lmjohns3/theano-nets
    cd theano-nets
    python setup.py develop

## Getting started

There are a few examples in the `examples/` directory. Run an example with the
`--help` flag to get a list of all the command-line arguments ; there are many
of them, but some of the notable ones are :

    -n or --layers N1 N2 N3 N4

Build a network with `N1` inputs, two hidden layers with `N2` and `N3` units,
and `N4` outputs. (Note that this argument is fixed in the code for the
examples, since it needs to correspond to the shape of the data being
processed.)

    -g or --activation logistic|relu|linear|norm:mean+logistic|...

Use the given activation function for hidden layer units. (All output layer
units have a linear activation function.) Several activation functions can be
pipelined together using `+`.

    -O or --optimize sgd|hf|sgd+hf|...

Use the given optimization method to train network parameters. Like the
activations, several training methods can be used in sequence by concatenating
their names with `+`.

## Using the library

Probably the easiest way to start with the library is to copy one of the
examples and modify it to perform your tasks. The usual workflow involves
instantiating `theanets.Experiment` with a subclass of `theanets.Network`, then
adding some data by calling `add_dataset(...)`, and finally calling `train()` to
learn a good set of parameters for your data. You can then `save()` the trained
model to a pickle, or call the trained `network` directly with new data to
compute a feedforward pass.

The documentation is relatively sparse, so please file bugs if you find that
there's a particularly hard-to-understand area in the code.
