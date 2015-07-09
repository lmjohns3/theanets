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

   build
   Loss
   MeanSquaredError
   MeanAbsoluteError
   CrossEntropy
   Hinge

Layer types
===========

.. automodule:: theanets.layers.base
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   build
   Layer

Feedforward layers
------------------

.. automodule:: theanets.layers.feedforward
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Classifier
   Feedforward
   Input
   Tied

Convolution layers
------------------

.. automodule:: theanets.layers.convolution
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Convolution
   Conv1

Recurrent layers
----------------

.. automodule:: theanets.layers.recurrent
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Recurrent
   RNN
   ARRNN
   LRRNN
   GRU
   LSTM
   Clockwork
   MRNN
   Bidirectional

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
