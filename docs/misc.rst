====================
Miscellaneous Topics
====================

Using the Command Line
======================

The ``theanets`` package was designed from the start to use the command line for
configuring most aspects of defining and training a model.

If you work in a command-line environment, you can leave many of the
hyperparameters for your model unspecified when constructing your
:class:`Experiment <theanets.main.Experiment>`, and instead specify the
configuration of your network using flags defined on the command line::

  exp = theanets.Experiment(theanets.Classifier)

This will create the same network as the classification model described above if
you run your file as::

  (venv)~$ mnist-classifier.py --layers 784 100 10

In both cases, the model has one input layer with 784 units, one hidden layer
containing 100 model neurons, and one softmax output layer with 10 units.

Command-line arguments can be stored in text files (one argument per line) and
loaded from the command-line using the ``@`` prefix::

  (venv)~$ mnist-classifier.py @args.txt

.. note::
   Command-line arguments do not work when running ``theanets`` code in IPython;
   within IPython, all parameters must be specified as keyword arguments.

You can set many more hyperparameters on the command line. Use the ``--help``
flag from the command line to show the options that are currently available.
