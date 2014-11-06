==================
Training a Network
==================

.. _trainers-sgd:

Trainers based on Stochastic Gradient Descent
=============================================

Stochastic Gradient Descent (sgd)
---------------------------------

--learning-rate
--momentum

Nesterov's Accelerated Gradient (nag)
-------------------------------------

--learning-rate
--momentum

Resilient Backpropagation (rprop)
---------------------------------

--learning-rate (sets initial learning rate)
--rprop-increase, --rprop-decrease
--rprop-min-step, --rprop-max-step

RMS-scaled Backpropagation (rmsprop)
------------------------------------

--learning-rate
--momentum (sets decay for exponential moving average)

.. _trainers-hf:

Second-Order Gradient Descent
=============================

Hessian-Free Method
-------------------

.. _trainers-scipy:

Trainers from Scipy
===================

``theanets`` incorporates a "Scipy" trainer that serves as a wrapper for the
optimization algorithms provided in ``\ `scipy.optimize.minimize`_\ ``. The
following algorithms are available in this trainer:

- bfgs
- cg
- dogleg
- newton-cg
- trust-ncg

In general, these methods require two types of computations in order to minimize
a cost function: evaluating the cost function for a specific setting of model
parameters, and computing the gradient of the cost function for a specific
setting of model parameters. Both of these computations are implemented by the
``theanets`` package and may, if you have a GPU, involve computing values on the
GPU.

However, all of the optimization steps that might be performed once these two
types of values are computed will not be handled on the GPU, since Scipy is
GPU-unaware. This might or might not influence the absolute time required to
optimize a model, depending on the ratio of time spent computing cost and
gradient values to the time spent computing parameter updates.

For more information about these optimization methods, please see the
`Scipy documentation`_.

.. _scipy.optimize.minimize: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. _Scipy documentation: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

.. _trainers-misc:

Other Training Techniques
=========================

Sampling (sample)
-----------------

One surprisingly effective training method for density estimation models is
simply to draw samples randomly from the training data and use those samples as
the "learned" features in the model.

Layerwise (layerwise)
---------------------

Because neural network models are arranged in :math:`k`-partite layers, they
admit to special optimization methods that are not easily described in terms of
taking gradient steps with respect to all of the parameters in the model. One of
the most effective methods for training networks that has been developed in the
past ten years or so has been so-called "greedy layerwise pre-training" methods.

In a layerwise pre-training paradigm, each of the layers of the network is
treated as a separate optimization problem, in a specific way.
