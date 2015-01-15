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

.. _qs-autoencoder:

Autoencoders
------------

The ``theanets`` package also provides an :class:`Autoencoder
<theanets.feedforward.Autoencoder>` class to construct models that can learn
features from data without labels. An autoencoder for MNIST digits, for example,
takes as input an unlabeled MNIST digit image and then attempts to produce this
same digit image as output. The hidden layers in such a model are then called
the "features" of the data that the model learns.

An autoencoder must always have the same number of inputs as outputs. The output
layer typically has a linear activation, which treats the data as a weighted sum
of some fixed set of *basis vectors* that spans the space of the data being
modeled. For an MNIST autoencoder task, your model must have 784 inputs and 784
outputs.

There can be any number of layers between the input and output, and they can be
of practically any form, but there are a few notable classes of autoencoders:

- *Undercomplete autoencoders* (also called *bottleneck autoencoders*) have a
  hidden layer that is smaller than the input layer. A small hidden layer is
  referred to as a bottleneck because the model must find some way to compress
  the input data into a smaller-dimensional space without losing too much
  information.

- *Overcomplete autoencoders* have hidden layers that are all larger than the
  input layer. These models are capable of learning a trivial identity transform
  from the inputs to the hidden layer(s) and on to the outputs, so they are
  often *regularized* in various ways to learn robust features.

  For example, a :ref:`sparse autoencoder <models-sparse-autoencoder>` is
  penalized for using large values in the hidden-unit activations, and a
  :ref:`denoising autoencoder <models-denoising-autoencoder>` adds noise to the
  inputs and forces the model to reconstruct the noise-free inputs.

- As with classifiers, *deep autoencoders* are any autoencoder model with more
  than a small number of hidden layers. Deep models have been quite popular
  recently, as they perform quite well on a variety of difficult machine
  learning tasks.

Finally, some autoencoders are capable of using *tied weights*, which means the
"input" weights are the same as the "output" weights in the model. Autoencoders
with tied weights represent some very common machine learning algorithms; see
:ref:`models-tied-weights` for more information.

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

Contributing
============

Fork the ``theanets`` code using git: https://github.com/lmjohns3/theanets
