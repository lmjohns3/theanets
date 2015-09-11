.. _trainers:

========
Trainers
========

The most common method for training a neural network model is to use a
stochastic gradient-based optimizer. In ``theanets`` many of these algorithms
are available by interfacing with the ``downhill`` package:

- ``sgd``: `Stochastic gradient descent`_
- ``nag``: `Nesterov's accelerated gradient`_
- ``rprop``: `Resilient backpropagation`_
- ``rmsprop``: RMSProp_
- ``adadelta``: ADADELTA_
- ``esgd``: `Equilibrated SGD`_
- ``adam``: Adam_

.. _Stochastic gradient descent: http://downhill.readthedocs.org/en/stable/generated/downhill.first_order.SGD.html
.. _Nesterov's accelerated gradient: http://downhill.readthedocs.org/en/stable/generated/downhill.first_order.NAG.html
.. _Resilient backpropagation: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.RProp.html
.. _RMSProp: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.RMSProp.html
.. _ADADELTA: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ADADELTA.html
.. _Equilibrated SGD: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.ESGD.html
.. _Adam: http://downhill.readthedocs.org/en/stable/generated/downhill.adaptive.Adam.html

In addition to the optimization algorithms provided by ``downhill``,
``theanets`` defines a few algorithms that are more specific to neural networks.
These trainers tend to take advantage of the layered structure of the loss
function for a network.

- ``sample``: :class:`Sample trainer <theanets.trainer.SampleTrainer>`

This trainer sets model parameters directly to samples drawn from the training
data. This is a very fast "training" algorithm since all updates take place at
once; however, often features derived directly from the training data require
further tuning to perform well.

- ``layerwise``: :class:`Layerwise (supervised) pretrainer <theanets.trainer.SupervisedPretrainer>`

Greedy supervised layerwise pre-training: This trainer applies RMSProp to each
layer sequentially.

- ``pretrain``: :class:`Unsupervised pretrainer <theanets.trainer.UnsupervisedPretrainer>`

Greedy unsupervised layerwise pre-training: This trainer applies RMSProp to a
tied-weights "shadow" autoencoder using an unlabeled dataset, and then transfers
the learned autoencoder weights to the model being trained.
