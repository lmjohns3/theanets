theanets
========

This package contains implementations of several common neural network
structures, using the amazing Theano_ package for optimization.

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
``--help`` flag to get a list of all the command-line arguments ; there are many
of them, but some of the notable ones are::

    -n or --layers N1 N2 N3 N4

Build a network with ``N1`` inputs, two hidden layers with ``N2`` and ``N3``
units, and ``N4`` outputs. (Note that this argument is held constant in the
example code, since it needs to correspond to the shape of the data being
processed.)

::
    -g or --activation logistic|relu|linear|norm:mean+logistic|...

Use the given activation function for hidden layer units. (All output layer
units have a linear activation function.) Several activation functions can be
pipelined together using whitespace.

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
then adding some data by calling ``add_dataset(...)``, and finally calling
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

License
-------

This package is distributed under an MIT license.

Copyright (c) 2013 Leif Johnson <leif@leifjohnson.net>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
