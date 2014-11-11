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

Background
==========

A feedforward neural network describes a parametric mapping

.. math::
   F_\theta: \mathcal{S} \to \mathcal{T}

between a source space :math:`\mathcal{S}` and a target space
:math:`\mathcal{T}`. For the MNIST digits, for example,
:math:`\mathcal{S} = \mathbb{R}^{784}`, and for a digit classification task,
:math:`\mathcal{T} = \mathbb{R}^{10}`.

.. tikz::
   :libs: arrows

   [thick,->,>=stealth',rectangle,minimum size=10mm,node distance=25mm,rounded corners=3mm]
   \node (dots) at (0, 0) {$\dots$};
   \node[draw] (h1) [left of=dots] {Layer 1} edge (dots);
   \node[draw] (input) [left of=h1] {Input} edge (h1);
   \node[draw] (hkm1) [right of=dots] {Layer $k-2$} edge[<-] (dots);
   \node[draw] (output) [right of=hkm1] {Output} edge[<-] (hkm1);

The mapping :math:`F_\theta` is implemented in neural networks by assuming a
specific, layered form. Computation nodes -- also called units or sometimes
neurons -- are arranged in a :math:`k` partite graph, with layer :math:`k`
containing :math:`n_k` nodes. A *weight* matrix :math:`W^k \in \mathbb{R}^{n_k
\times n_{k+1}}` specifies the strength of the connection between nodes in layer
:math:`k` and those in layer :math:`k+1` -- all other pairs of nodes are not
connected. Each node typically also has a scalar *bias* parameter that
determines its offset from the origin. Together, the parameters :math:`\theta`
of the model are these :math:`k-1` weight matrices and :math:`k` bias vectors.

.. tikz::
   :libs: arrows

   [thick,->,>=stealth',circle,minimum size=10mm,node distance=10mm,below]
   \node[draw] (x) at (0, 0) {$a_i^{k+1}$};
   \node[draw] (b) at (-30mm, 0) {$a_j^k$} edge node {$w^k_{ji}$} (x);
   \node (adots) [above of=b] {$\vdots$};
   \node[draw] (a) [above of=adots] {$a_1^k$} edge node[above] {$w^k_{1i}$} (x);
   \node (cdots) [below of=b] {$\vdots$};
   \node[draw] (c) [below of=cdots] {$a_{n_k}^k$} edge node {$w^k_{n_ki}$} (x);

In a standard feedforward network, each node :math:`i` in layer :math:`k+1`
receives inputs from all nodes in layer :math:`k`, then transforms the weighted
sum of these inputs:

.. math::
   a_i^{k+1} = \sigma\left( \sum_{j=1}^{n_k} w^k_{ij} a_j^k \right)

where :math:`\sigma: \mathbb{R} \to \mathbb{R}` is some activation function.
Typical choices of the activation function are linear :math:`\sigma(z) = z`,
rectified :math:`\sigma(z) = \max(0, z)`, or sigmoidal :math:`\sigma(z) = (1 +
e^{-z})^{-1}`. Usually all hidden nodes in a network share the same activation
function; nodes in the input layer are assumed to have linear activation, and
nodes in the output layer might have linear or nonlinear activations depending
on the modeling task.

.. _models-autoencoders:

Autoencoders
============

An autoencoder defines a mapping from a source space to itself.

.. math::
   F_\theta: \mathcal{S} \to \mathcal{S}

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
   \ell = \frac{1}{M} \sum_{i=1}^M \| F_\theta(x_i) - y_i \|_2^2 + R(\theta)

.. _models-classification:

Classification
==============

.. math::
   \ell = \frac{1}{M} \sum_{i=1}^M \| F_\theta(x_i) - y_i \|_2^2 + R(\theta)

.. _models-regularization:

Regularization
==============

One heuristic that can prevent parameters from overtraining on small datasets is
based on the observation that "good" parameter values are typically small: large
parameter values often indicate overfitting. One way to encourage a model to use
small parameter values is to assume that the parameter values are sampled from a
posterior distribution over parameters, conditioned on observed data. In this
way of thinking about parameters, we can manipulate the prior distribution of
the parameter values to express our knowledge as modelers of the problem at
hand.

Decay
-----

In "weight decay," we assume that parameters are drawn from a zero-mean Gaussian
distribution with an isotropic, modeler-specified standard deviation. In terms
of loss functions, this equates to adding a term to the loss function that
computes the :math:`L_2` norm of the parameter values in the model:

.. math::
   \ell = \dots + \frac{\lambda}{2} \| \theta \|_2^2

If the loss :math:`\ell` represents some approximation to the log-posterior
distribution of the model parameters given the data

.. math::
   \ell = \log p(\theta|x) \propto \dots + \frac{\lambda}{2} \| \theta \|_2^2

then the term with the :math:`L_2` norm on the parameters is like an unscaled
Gaussian distribution.

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
