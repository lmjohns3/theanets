==============
Network Models
==============

``theanets`` comes with several families of popular neural network models
built-in. This document describes the available models using the language of
mathematical optimization, and then shows how the ``theanets`` code needs to be
invoked to use these models.

Background
==========

A feedforward neural network describes a parametric mapping

.. math::
   r_\theta: \mathcal{S} \to \mathcal{T}

between a source space :math:`\mathcal{S}` and a target space
:math:`\mathcal{T}`. The parameters :math:`\theta` describe all trainable
elements of the mapping, primarily consisting of weights connecting computation
nodes, and biases for the computation nodes.

.. tikz::
    \draw[thick,rounded corners=8pt]
    (0,0)--(0,2)--(1,3.25)--(2,2)--(2,0)--(0,2)--(2,2)--(0,0)--(2,0);

Neural networks actually compute an approximation to :math:`r` by assuming that
the computations conform to a specific kind of structure. In particular, a
network with :math:`k` layers defines a :math:`k`-partite graph of computational
nodes that are often called units or neurons. The nodes in this graph are just
convenient ways of representing specific types of computations, and the edges in
the graph describe the pathways along which information can flow from one node
to another.

Traditionally, each node :math:`i` in layer :math:`k+1` receives inputs from all
nodes in layer :math:`k` and performs a transform of a weighted summation of
these inputs:

.. math::
   a_i^{k+1} = \sigma\left( \sum_{j=1}^{n_k} w_{ij} a_j^k \right)

where :math:`\sigma(\cdot)` is some activation function, typically linear
:math:`\sigma(z) = z`, rectified :math:`\sigma(z) = \max(0, z)`, or sigmoidal
:math:`\sigma(z) = (1 + e^{-z})^{-1}`.

.. _models-autoencoders:

Autoencoders
============

An autoencoder defines a mapping from a source space to itself.

.. math::
   r_\theta: \mathcal{S} \to \mathcal{S}

Typically, this mapping is decomposed into an "encoding" :math:`f_\alpha(\cdot)`
and a corresponding "decoding" :math:`g_\beta(\cdot)` to and from a latent space
:math:`\mathcal{T}`:

.. math::
   f_\alpha: \mathcal{S} \to \mathcal{T}, \qquad
   g_\beta: \mathcal{T} \to \mathcal{S}

Autoencoders are often trained with respect to a reconstruction loss:

.. math::
   \ell = \frac{1}{M} \sum_{i=1}^M \left\| g_\beta\left(f_\alpha(x_i)\right) - x_i \right\|_2^2 + R(\alpha, \beta)

The idea here is that the model should be able to recover the original input
data after performing the encoding and subsequent decoding of the input. Seen
from this perspective, the autoencoder is learning an encoding that preserves as
much information as possible about the input (in the least-squares sense).

Sparse autoencoders
-------------------

A sparse autoencoder assigns a regularization penalty to the hidden activation
of the model.

.. math::
   \ell = \frac{1}{M} \sum_{i=1}^M \left\| g_\beta\left(f_\alpha(x_i)\right) - x_i \right\|_2^2 + \lambda\left\| f_\alpha(x_i) \right\|_1

This penalty forces the encoder and decoder of the autoencoder model to
cooperate together to represent the input using as little of the latent space as
possible.

.. _models-regression:

Regression
==========

.. math::
   \ell = \frac{1}{M} \sum_{i=1}^M \| r_\theta(x_i) - y_i \|_2^2 + R(\theta)

.. _models-classification:

Classification
==============

.. math::
   \ell = \frac{1}{M} \sum_{i=1}^M \| r_\theta(x_i) - y_i \|_2^2 + R(\theta)

.. _models-regularization:

Regularization
==============

Sparsity
--------

Sparse models have been shown to capture regularities seen in the mammalian
visual cortex [3]_. In addition, sparse models in machine learning are often
more performant than "dense" models without restriction on the hidden
representation [1]_. Furthermore, sparse models tend to yield latent
representations that are more interpretable to humans than dense models [2]_.

References
==========

.. [1] Lee et al, "Sparse representation"
.. [2] Tibshirani, "Lasso"
.. [3] Olshausen, B and Field, DJ.
