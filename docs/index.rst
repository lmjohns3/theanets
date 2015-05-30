============
``THEANETS``
============

The ``theanets`` package provides tools for defining and optimizing several
common types of neural network models. It uses Python for rapid development, and
under the hood Theano_ provides graph optimization and fast computations on the
GPU.

The package defines models for
:class:`classification <theanets.feedforward.Classifier>`,
:class:`autoencoding <theanets.feedforward.Autoencoder>`,
:class:`regression <theanets.feedforward.Regressor>`, and
:class:`prediction <theanets.recurrent.Predictor>`. Models can easily be created
with any number of :class:`feedforward <theanets.layers.Feedforward>` or
:class:`recurrent <theanets.layers.Recurrent>` :mod:`layers <theanets.layers>`
and combined with different regularizers:

- L1/L2 weight decay
- L1/L2 hidden activation penalties (e.g., sparse autoencoders)
- Dropout/gaussian noise (e.g., denoising autoencoders)

Several optimization algorithms are also included:

- :class:`SGD <theanets.trainer.SGD>` and variants:
  :class:`NAG <theanets.trainer.NAG>`,
  :class:`Rprop <theanets.trainer.Rprop>`,
  :class:`RmsProp <theanets.trainer.RmsProp>`,
  :class:`ADADELTA <theanets.trainer.ADADELTA>`
- Many algorithms from ``scipy.optimize.minimize``
- Greedy :class:`layerwise <theanets.trainer.Layerwise>` pre-training

The source code for ``theanets`` lives at http://github.com/lmjohns3/theanets,
the documentation lives at http://theanets.readthedocs.org, and announcements
and discussion happen on the `mailing list`_.

.. _Theano: http://deeplearning.net/software/theano/
.. _mailing list: https://groups.google.com/forum/#!forum/theanets

Example Code
============

Let's say you wanted to create a classifier and train it on some 100-dimensional
data points that you've classified into 10 categories. You can define your model
and train it using a few lines of code::

  import climate
  import sklearn.datasets
  import sklearn.metrics
  import theanets

  climate.enable_default_logging()

  X, y = sklearn.datasets.make_classification(
      n_samples=3000,
      n_features=100,
      n_informative=30,
      n_repeated=10,
      n_redundant=10,
      n_classes=10,
      n_clusters_per_class=3,
  )
  X = X.astype('f')
  y = y.astype('i')

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(100, 200, 100, 10),
  )

  cut = 2500
  exp.train(
      [X[:cut], y[:cut]],
      [X[cut:], y[cut:]],
      algo='sgd',
      learning_rate=0.01,
      momentum=0.5,
      hidden_l1=0.001,
      weight_l2=0.001,
  )

  print('training:')
  print(sklearn.metrics.confusion_matrix(
      y[:cut], exp.network.predict(X[:cut])))

  print('validation:')
  print(sklearn.metrics.confusion_matrix(
      y[cut:], exp.network.predict(X[cut:])))

The remainder of the documentation will help fill you in on the details of these
calls and the options that ``theanets`` provides for each of them. Have fun!

Documentation
=============

.. toctree::
   :maxdepth: 2

   quickstart
   creating
   training
   using
   misc
   reference
