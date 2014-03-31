:mod:`theanets`

Quickstart
==========

This page provides a quick overview of the ``theanets`` package that should get
you up and running with a few simple examples. Once you understand the basic
workflow, you should be able to extend the examples to your own datasets! Along
the way, there are links to the reference documentation and to the User's Guide,
which expands on the ideas here to hopefully make them more accessible.

If you find an error in these documents, or just want to clarify something,
please send a pull request to https://github.com/lmjohns3/theano-nets and we'll
work on fixing it up!

Basic overview
--------------

At its core, ``theanets`` is a tool for optimization. It helps you define,
compute and optimize complex, parametric functions that map a set of input
variables to a set of output variables. The vocabulary of neural networks is
used throughout the package because it's convenient and expressive, but
optimization is the name of the game.

What does this mean for you? Suppose you have some image data, and you'd like to
build a model that tells you when some image depicts an elephant. Well, you can
imagine that your image data represents measurements of a (possibly very large)
set of variables, and the fact that there is an elephant (or not) in your images
represents another variable. You can use the ``theanets`` package to define a
parametric function for doing this task; the result of this definition is called
the "model" or the "architecture." Your parametric function will take as input
an image and yield as output a prediction of whether that image depicts an
elephant! (This part is called the "feedforward computation.") Along the way
you'll need to find the parameter settings that best accomplish this task; this
is called "training" or "optimization," and is accomplished using
"[backpropagation]_," or, more generally, "gradient descent."

.. _backpropagation: http://en.wikipedia.org/wiki/Backpropagation

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

