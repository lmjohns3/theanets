==============
Network Models
==============

``theanets`` comes with several families of popular neural network models
built-in. This page describes the available models using the language of
mathematical optimization, and then shows how the ``theanets`` code can be
invoked to use these models.

The examples throughout the documentation use the `MNIST digits dataset
<http://yann.lecun.com/exdb/mnist/>`_, a set of 70,000 28×28 images of
hand-written digits. Each MNIST digit is labeled with the correct digit class
(0, 1, ... 9). Please see the :ref:`qs-mnist` section for a brief overview of
this dataset if you're not already familiar with it.

.. _models-background:

Mathematics Background
======================

Datasets
--------

Loss function
-------------

Having created a model and collected some data, we must close by defining a
*loss* that our model is expected to minimize in order to perform well on a
task. Many types of models use a *squared-error loss*, but other losses such as
*cross-entropy* are also useful.

For an autoencoder, which only receives an "input" dataset :math:`X` (remember
it has :math:`M` rows), the squared-error loss encourages the model to
reconstruct its input:

.. math::
   J(X, \theta) = \frac{1}{M} \sum_{i=1}^M \left\| F_\theta(x_i) - x_i \right\|_2^2 + R(X, \theta)

For a regression model, which also receives a target output dataset :math:`K`,
the squared-error loss encourages the model to match the target:

.. math::
   J(X, K, \theta) = \frac{1}{M} \sum_{i=1}^M \left\| F_\theta(x_i) - k_i \right\|_2^2 + R(X, \theta)

In both of these examples, an additional *regularization* term :math:`R(\cdot)`
is added to the loss; it is typically some function of the dataset and the
parameters. This regularizer can be chosen to encourage different types of model
behavior, often to reflect different types of prior assumptions that the modeler
has about the problem at hand.

Having defined a loss for a model, the best parameters are those that minimize
the loss on the data that we have:

.. math::
   \theta = \arg\min_\Omega J(\cdot, \Omega)

For some classes of models, this optimization procedure is quite straightforward
and even has close-form solutions. For many classes of neural network models,
however, this optimization procedure is quite tricky. See :doc:`trainers` for
more information about optimization.

