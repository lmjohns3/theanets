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

Mathematics Background
======================

Neural networks are really just a concise, computational way of describing a
mathematical model of some data. Before getting into the models below, we'll
first set up the ideas and notation that are used on this page.

At a high level, a feedforward neural network describes a parametric mapping

.. math::
   F_\theta: \mathcal{S} \to \mathcal{T}

between a source space :math:`\mathcal{S}` and a target space
:math:`\mathcal{T}`. For the MNIST digits, for example we could think of
:math:`\mathcal{S} = \mathbb{R}^{28 \times 28} = \mathbb{R}^{784}` (i.e., the
space of all 28×28 images), and for classifying the MNIST digits we could think
of :math:`\mathcal{T} = \{0, 1\}^{10}` (i.e., vectors of length 10 composed only
of 0s and 1s).

This mapping is assumed to be fairly complex. If it were not -- if you could
capture the mapping using a simple expression like :math:`F_\theta(x) = x^2` --
then we would just use the expression directly and not need to deal with an
entire network. So if the mapping is complex, we will do a couple of things to
make our problem tractable. First, we will assume some structure for
:math:`F_\theta`. Second, we will fit our model to some set of data that we have
obtained.

Model structure
---------------

.. tikz::
   :libs: arrows

   [thick,->,>=stealth',rectangle,minimum size=10mm,node distance=25mm,rounded corners=3mm]
   \node (dots) at (0, 0) {$\dots$};
   \node[draw] (h1) [left of=dots] {Layer 1} edge (dots);
   \node[draw] (input) [left of=h1] {Input} edge (h1);
   \node[draw] (hkm1) [right of=dots] {Layer $k-2$} edge[<-] (dots);
   \node[draw] (output) [right of=hkm1] {Output} edge[<-] (hkm1);

The mapping :math:`F_\theta` is implemented in neural networks by assuming a
specific, layered form. Computation nodes -- also called units or (sometimes)
neurons -- are arranged in a :math:`k` partite graph, with layer :math:`k`
containing :math:`n_k` nodes. A **weight matrix** :math:`W^k \in \mathbb{R}^{n_k
\times n_{k+1}}` specifies the strength of the connection between nodes in layer
:math:`k` and those in layer :math:`k+1` -- all other pairs of nodes are not
connected. Each layer of nodes typically also has a **bias vector** that
determines the offset of each node from the origin. Together, the parameters
:math:`\theta` of the model are these :math:`k-1` weight matrices (weights only
exist between layers of computation nodes) and :math:`k-1` bias vectors (there
are no biases for the input nodes in the graph).

.. tikz::
   :libs: arrows

   [thick,->,>=stealth',circle,minimum size=10mm,node distance=10mm,below,near start]
   \node[draw] (z) at (0, 0) {$\sum$};
   \node[draw] (x) at (20mm, 1.5mm) {$a_i^{k+1}$} edge[<-] (z);
   \node[draw] (b) at (-30mm, 0) {$a_j^k$} edge node {$w^k_{ji}$} (z);
   \node (adots) [above of=b] {$\vdots$};
   \node[draw] (a) [above of=adots] {$a_1^k$} edge node {$w^k_{1i}$} (z);
   \node (cdots) [below of=b] {$\vdots$};
   \node[draw] (c) [below of=cdots] {$a_{n_k}^k$} edge node {$w^k_{n_ki}$} (z);
   \node[draw] (bias) at (0, -20mm) {$b^{k+1}_i$} edge (z);

In a standard feedforward network, each node :math:`i` in layer :math:`k+1`
receives inputs from all nodes in layer :math:`k`, then transforms the weighted
sum of these inputs:

.. math::
   a_i^{k+1} = \sigma\left( b_i^{k+1} + \sum_{j=1}^{n_k} w^k_{ij} a_j^k \right)

where :math:`\sigma: \mathbb{R} \to \mathbb{R}` is an "activation function."
Although many functions will work, typical choices of the activation function
are:

- **linear** :math:`\sigma(z) = z`,
- **rectified/rectified linear/relu** :math:`\sigma(z) = \max(0, z)`, or
- **logistic sigmoid** :math:`\sigma(z) = (1 + e^{-z})^{-1}`.

Usually all hidden nodes in a network share the same activation function. Nodes
in the input layer are assumed to have linear activation (i.e., the input nodes
simply represent the state of the input data), and nodes in the output layer
might have linear or nonlinear activations depending on the modeling task.

Data
----

Now that we have defined some structure for the model, we will probably need to
fit the model to some empirical data. We assume that we have obtained a set of
samples from a distribution :math:`p(\mathcal{S}, \mathcal{T})` over the two
spaces of our model or problem.

The data that we need will depend on the task at hand. For a classification task
using the MNIST digits, for example, we will need some samples from
:math:`\mathcal{S} = \mathbb{R}^{28\times 28}` (e.g., pixel arrays of MNIST
digits) as well as the accompanying **labels** from :math:`\mathcal{T} = \{0,
1\}^{10}` digit labels (e.g., vectors with ten entries containing a 1 in the
slot representing the correct digit class). For an autoencoder or density
estimation task, we only need the **unlabeled** samples from
:math:`\mathcal{S}`.

The samples from :math:`\mathcal{S}` will be referred to below as :math:`x`,
while the samples (labels) from :math:`\mathcal{T}` will be referred to below as
:math:`y`. If many samples are grouped together, we'll assume they are rows in a
matrix :math:`X` or :math:`Y`.

.. note::
   Typically in mathematics samples are treated as columns, but the ``theanets``
   library, as well as many other Python-based machine learning libraries,
   treats these quantities as rows. To avoid confusion with the coding world,
   the math on this page assumes row vectors and row-oriented matrices.

With the mathematics notation out of the way, it's time to look at different
neural network models!

.. _models-autoencoders:

Autoencoders
============

An autoencoder defines a mapping from a source space to itself.

.. math::
   F_\theta: \mathcal{S} \to \mathcal{S}

Typically, this mapping is decomposed into an "encoding" stage
:math:`f_\alpha(\cdot)` and a corresponding "decoding" stage
:math:`g_\beta(\cdot)` to and from some latent space :math:`\mathcal{Z} =
\mathbb{R}^{n_z}`:

.. math::
   f_\alpha: \mathcal{S} \to \mathcal{Z}, \qquad
   g_\beta: \mathcal{Z} \to \mathcal{S}

Autoencoders form an interesting class of models for several reasons. They:

- require only "unlabeled" data (which is typically easy to obtain),
- are generalizations of many popular density estimation techniques, and
- can be used to model the "manifold" or density of a dataset.

A generic autoencoder can be defined in ``theanets`` by using the
:class:`theanets.Autoencoder` class::

  exp = theanets.Experiment(theanets.Autoencoder)

The ``layers`` parameter is required to define such a model; it can be provided
on the command-line by using ``--layers A B C ... A``, or in your code::

  exp = theanets.Experiment(
      theanets.Autoencoder,
      layers=(A, B, C, ..., A))

.. note::
   Command-line arguments do not work when running ``theanets`` code in IPython;
   within IPython, all parameters must be specified as keyword arguments.

A subset of autoencoders with an odd-length, palindromic number of layers can be
defined as having **tied weights** whenever the parameters from the decoder are
the transpose of the parameters from the encoder. Tied-weights autoencoders form
an interesting subset of autoencoder models.

Let's look at a few example models that fall into the autoencoder class.

Principal Component Analysis (PCA)
----------------------------------

The most popular density estimation technique out there is Principal Component
Analysis (PCA). Principal components are a set of orthogonal directions of
maximal variance in a dataset; that is, PCA computes a rotation of the dataset
such that the axes of the rotated system capture as much variance as possible.

In PCA, a modeler assumes that the data come from a single
ellipsoidal blob in :math:`\mathcal{S}`; that is, PCA assumes the data are drawn
from a multivariate Gaussian distribution :math:`p(X) = \mathcal{N}(\mu,
\Sigma)` and then automatically identifies the subspace :math:`\mathcal{Z} =
\mathbb{R}^{n_z}` such that a linear projection to :math:`\mathcal{Z}` and back
to :math:`\mathcal{S}` preserves the maximum variance in the data.

Let's look at this mathematically for a minute.

.. math::
   x = Wz + \epsilon = WW^\top x + \epsilon

   0 = WW^\top x - x + \epsilon

   -\log p(X) \propto \|WW^\top x - x\|_2^2

Given this way of looking at PCA, we can see that it is really a sort of linear
autoencoder with tied weights. To implement such a model in ``theanets``, we
only need to provide the following hyperparameters::

  pca = theanets.Experiment(
      theanets.Autoencoder,
      tied_weights=True,
      hidden_activation='linear',
  )

Sparse autoencoders
-------------------

A sparse autoencoder assigns a regularization penalty to the hidden activation
of the model.

.. math::
   \ell = \frac{1}{M} \sum_{i=1}^M \left\| g_\beta\left(f_\alpha(x_i)\right) - x_i \right\|_2^2 + \lambda\left\| f_\alpha(x_i) \right\|_1

This penalty forces the encoder and decoder of the autoencoder model to
cooperate together to represent the input using as little of the latent space as
possible.

Independent Component Analysis (ICA)
------------------------------------

While PCA assumes that the underlying data distribution is Gaussian, this
assumption is not necessarily true for many datasets. A better model for some
datasets like photographs of the natural world turns out to assume that the
underlying data distribution is *not* Gaussian, by maximizing the independence
of the latent components of the model.

One way to accomplish this maximization is to maximize the kurtosis of the model
distribution, but another is to force the model to use a sparse representation
while still using linear encoding and decoding with tied weights::

  ica = theanets.Experiment(
      theanets.Autoencoder,
      tied_weights=True,
      hidden_activation='linear',
      hidden_l1=1,
  )

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
