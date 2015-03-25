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

.. _Theano: http://deeplearning.net/software/theano/

.. _classification: http://theanets.readthedocs.org/en/stable/generated/theanets.feedforward.Classifier.html
.. _autoencoding: http://theanets.readthedocs.org/en/stable/generated/theanets.feedforward.Autoencoder.html
.. _regression: http://theanets.readthedocs.org/en/stable/generated/theanets.feedforward.Regressor.html
.. _prediction: http://theanets.readthedocs.org/en/stable/generated/theanets.recurrent.Predictor.html

.. _feedforward: http://theanets.readthedocs.org/en/stable/generated/theanets.layers.Feedforward.html
.. _recurrent: http://theanets.readthedocs.org/en/stable/generated/theanets.layers.Recurrent.html
.. _layers: http://theanets.readthedocs.org/en/stable/reference.html#module-theanets.layers

.. _SGD: http://theanets.readthedocs.org/en/stable/generated/theanets.trainer.SGD.html
.. _NAG: http://theanets.readthedocs.org/en/stable/generated/theanets.trainer.NAG.html
.. _Rprop: http://theanets.readthedocs.org/en/stable/generated/theanets.trainer.Rprop.html
.. _RmsProp: http://theanets.readthedocs.org/en/stable/generated/theanets.trainer.RmsProp.html
.. _ADADELTA: http://theanets.readthedocs.org/en/stable/generated/theanets.trainer.ADADELTA.html
.. _layerwise: http://theanets.readthedocs.org/en/stable/training.html#layerwise-pretraining

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

  import climate
  import theanets
  import my_data_set

  climate.enable_default_logging()

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(100, 200, 100, 10),
  )

  exp.train(
      my_data_set.training_data,
      my_data_set.validation_data,
      optimize='sgd',
      learning_rate=0.01,
      momentum=0.5,
      hidden_l1=0.1,
  )

  exp.network.predict(my_data_set.test_data)

The trained ``network`` lets you make predictions on unseen test data.

More Information
----------------

Source: https://github.com/lmjohns3/theanets

Documentation: http://theanets.readthedocs.org

Mailing list: https://groups.google.com/forum/#!forum/theanets
