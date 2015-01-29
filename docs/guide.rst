==========
User Guide
==========

.. _guide-existing:

Using Existing Models
=====================

Several broad classes of models are pre-defined in ``theanets``:

- :ref:`Classifier <models-classification>`: A model that maps its input onto a
  (usually small) set of output nodes that represent the probability of a label
  given the input.
- :ref:`Autoencoder <models-autoencoders>`: A model that attempts to reproduce
  its input as its output.
- :ref:`Regressor <models-regression>`: Like the classifier, but instead of
  attempting to produce a `one-hot`_ output label, a regressor attempts to
  produce some continuous-valued target vector for each input.

.. _one-hot: http://en.wikipedia.org/wiki/One-hot

It's also pretty simple to create custom models using ``theanets``; see
:ref:`hacking-extending` for more information.

.. _guide-model-hyperparameters:

Model Hyperparameters
=====================

By default, layers in ``theano`` are constructed using straightforward
:class:`feedforward <theanets.layers.Feedforward>` layers; these layers compute
a weighted (affine) transformation of their input, and then perform a point-wise
(i.e., independent on each computation unit) nonlinear transform.

Activation functions
--------------------

- linear
- logistic sigmoid
- hyperbolic tangent
- rectified linear
- softplus
- softmax

Regularizers
------------

If you want to set up a more sophisticated model like a classifier with sparse
hidden representations, you can add regularization hyperparameters when you
create your experiment::

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(784, 1000, 784),
      hidden_l1=0.1)

Here we've specified that our model has a single, overcomplete hidden layer, and
the activity of the hidden units in the network will be penalized with a 0.1
coefficient.

.. _guide-training-hyperparameters:

Training Hyperparameters
========================

Training as iteration
---------------------

The :func:`Experiment.train() <theanets.main.Experiment.train>` method is
actually just a thin wrapper over the underlying :func:`Experiment.itertrain()
<theanets.main.Experiment.itertrain>` method, which you can use directly if you
want to do something special during training::

  for monitors in exp.itertrain(train, valid, **kwargs):
      print(monitors['loss'])

Trainers yield a dictionary after each training iteration. The keys and values
in each dictionary give the costs and monitors that are computed during
training, which will vary depending on the model and the training algorithm.
However, there will always be a ``'loss'`` key that gives the value of the loss
function that is being optimized. For classifier models, the dictionary will
also have an ``'acc'`` key, which gives the percent accuracy of the classifier
model.

.. note::
   The :class:`HF <theanets.trainer.HF>` trainer and the :class:`Sample
   <theanets.trainer.Sample>` trainer always return loss values equal to -1.

.. _guide-extending:

Creating New Models
===================

.. _guide-extending-regularizers:

Defining Custom Regularizers
----------------------------

.. _guide-extending-costs:

Defining Custom Cost Functions
------------------------------

It's pretty straightforward to create models in ``theanets`` that use cost
functions that are different from the predefined :class:`Classifier
<theanets.feedforward.Classifier>` (which uses binary cross-entropy) and
:class:`Regressor <theanets.feedforward.Regressor>` (which uses mean squared
error). To define by a model with a new cost function, just create a new
subclass and override the ``cost`` property on your subclass. For example, to
create a regression model that uses mean absolute error::

  class MaeRegressor(theanets.Regressor):
      @property
      def cost(self):
          err = self.outputs[-1] - self.targets
          return TT.mean(abs(err).sum(axis=1))

Your cost function must return a theano expression that reflects the cost for
your model.

.. _guide-data:

Providing Data
==============

.. _guide-data-callables:

Using callables
---------------

You can provide a callable for a dataset. This callable must take no arguments
and must return a ``numpy`` array of the proper shape for your model.

For example, this code defines a ``batch()`` helper that chooses a random
dataset and a random offset for each batch::

  SOURCES = 'foo.npy', 'bar.npy', 'baz.npy'
  BATCH_SIZE = 64

  def batch():
      X = np.load(np.random.choice(SOURCES), mmap_mode='r')
      i = np.random.randint(len(X))
      return X[i:i+BATCH_SIZE]

  # ...

  exp.train(batch)

If you need to maintain more state than is reasonable from a single closure, you
can also encapsulate the callable inside a class. Just make sure instances of
the class are callable by defining the ``__call__`` method::

  class Loader:
      def __init__(sources=('foo.npy', 'bar.npy', 'baz.npy'), batch_size=64):
          self.sources = sources
          self.batch_size = batch_size
          self.src = -1
          self.idx = 0
          self.X = ()

      def __call__(self):
          if self.idx + self.batch_size > len(self.X):
              self.idx = 0
              self.src = (self.src + 1) % len(self.sources)
              self.X = np.load(self.sources[self.src], mmap_mode='r')
          try:
              return self.X[self.idx:self.idx+self.batch_size]
          finally:
              self.idx += self.batch_size

  # ...

  exp.train(Loader())

.. _guide-contributing:

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

More Information
================

This concludes the user guide! You can read more information about ``theanets``
in the :doc:`quickstart` and :doc:`reference` sections of the documentation.

The source code for ``theanets`` lives at http://github.com/lmjohns3/theanets.
Please fork, explore, and send pull requests!

Finally, there is also a mailing list for project discussion and announcements.
Subscribe online at https://groups.google.com/forum/#!forum/theanets.
