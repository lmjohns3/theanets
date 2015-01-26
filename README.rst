theanets
========

The ``theanets`` package provides tools for defining and optimizing several
common types of neural network models. It uses Python for rapid development, and
under the hood Theano_ provides graph optimization and fast computations on the
GPU.

The package defines models for classification_, autoencoding_, regression_, and
prediction_. Models can easily be created with any number of feedforward_ or
recurrent_  layers_ and combined with different regularizers:

- L1/L2 weight decay
- L1/L2 hidden activation penalties (e.g., sparse autoencoders)
- Dropout/gaussian noise (e.g., denoising autoencoders)

Several optimization algorithms are also included:

- SGD_ and variants: NAG_, Rprop_, RmsProp_, ADADELTA_
- Many algorithms from ``scipy.optimize.minimize``
- Greedy layerwise_ pre-training

At present there are no RBMs, convolutions, or maxout in ``theanets`` -- for
those, you might want to look at Morb_, Lasagne_, or pylearn2_. There are many
other neural networks toolkits out there as well, in many other languages; see
`this stackoverflow question`_ for a few additional pointers, or just search for
them.

.. _Theano: http://deeplearning.net/software/theano/

.. _classification: http://theanets.readthedocs.org/en/latest/generated/theanets.feedforward.Classifier.html
.. _autoencoding: http://theanets.readthedocs.org/en/latest/generated/theanets.feedforward.Autoencoder.html
.. _regression: http://theanets.readthedocs.org/en/latest/generated/theanets.feedforward.Regressor.html
.. _prediction: http://theanets.readthedocs.org/en/latest/generated/theanets.recurrent.Predictor.html

.. _feedforward: http://theanets.readthedocs.org/en/latest/generated/theanets.layers.Feedforward.html
.. _recurrent: http://theanets.readthedocs.org/en/latest/generated/theanets.layers.Recurrent.html
.. _layers: http://theanets.readthedocs.org/en/latest/reference.html#module-theanets.layers

.. _SGD: http://theanets.readthedocs.org/en/latest/generated/theanets.trainer.SGD.html
.. _NAG: http://theanets.readthedocs.org/en/latest/generated/theanets.trainer.NAG.html
.. _Rprop: http://theanets.readthedocs.org/en/latest/generated/theanets.trainer.Rprop.html
.. _RmsProp: http://theanets.readthedocs.org/en/latest/generated/theanets.trainer.RmsProp.html
.. _ADADELTA: http://theanets.readthedocs.org/en/latest/generated/theanets.trainer.ADADELTA.html
.. _layerwise: http://theanets.readthedocs.org/en/latest/generated/theanets.trainer.Layerwise.html

.. _Morb: https://github.com/benanne/morb
.. _Lasagne: https://github.com/benanne/Lasagne
.. _pylearn2: http://deeplearning.net/software/pylearn2
.. _this stackoverflow question: http://stackoverflow.com/questions/11477145/open-source-neural-network-library

Installation
------------

Install the latest published code using pip::

    pip install theanets

Or download the current source and run it from there::

    git clone http://github.com/lmjohns3/theanets
    cd theanets
    python setup.py develop

Example
-------

Let's say you wanted to create a classifier and train it on some 100-dimensional
data points that you've classified into 10 categories. You can define your model
and train it using a few lines of code::

  import theanets
  import my_data_set

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(100, 200, 100, 10),
      hidden_l1=0.1,
  )

  exp.train(
      my_data_set.training_data,
      my_data_set.validation_data,
      optimize='sgd',
      learning_rate=0.01,
      momentum=0.5,
  )

The remainder of the documentation will help fill you in on the details of these
calls and the options that ``theanets`` provides for each of them. Have fun!

More Information
----------------

Source: https://github.com/lmjohns3/theanets

Documentation: http://theanets.readthedocs.org

Mailing list: https://groups.google.com/forum/#!forum/theanets
