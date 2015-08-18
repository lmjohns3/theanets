=========
Reference
=========

Computation graphs
==================

.. automodule:: theanets.graph
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Network

Feedforward networks
====================

.. automodule:: theanets.feedforward
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Autoencoder
   Classifier
   Regressor

Recurrent networks
==================

.. automodule:: theanets.recurrent
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Autoencoder
   Classifier
   Regressor

Recurrent helpers
-----------------

.. autosummary::
   :toctree: generated/

   batches
   Text

Losses
======

.. automodule:: theanets.losses
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Loss
   CrossEntropy
   GaussianLogLikelihood
   Hinge
   KullbackLeiblerDivergence
   MaximumMeanDiscrepancy
   MeanAbsoluteError
   MeanSquaredError

Regularizers
------------

.. automodule:: theanets.regularizers
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Regularizer
   HiddenL1
   WeightL1
   WeightL2
   Contractive
   Dropout
   GaussianNoise

Layer types
===========

.. automodule:: theanets.layers.base
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Concatenate
   Flatten
   Input
   Layer
   Product
   Reshape

Feedforward layers
------------------

.. automodule:: theanets.layers.feedforward
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Classifier
   Feedforward
   Tied

Convolution layers
------------------

.. automodule:: theanets.layers.convolution
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Conv1

Recurrent layers
----------------

.. automodule:: theanets.layers.recurrent
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   ARRNN
   Bidirectional
   Clockwork
   GRU
   LRRNN
   LSTM
   MRNN
   MUT1
   RNN

Activations
===========

.. automodule:: theanets.activations
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   build
   Activation
   Prelu
   LGrelu
   Maxout

Training strategies
===================

.. automodule:: theanets.trainer
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   DownhillTrainer
   SampleTrainer
   SupervisedPretrainer
   UnsupervisedPretrainer
