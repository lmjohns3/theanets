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

Neural networks are really just a concise, computational way of describing a
mathematical model of some data. Before getting into the models below, we'll
first set up the ideas and notation that are used on this page.

At a high level, a feedforward neural network describes a parametric mapping

.. math::
   F_\theta: \mathcal{S} \to \mathcal{T}

between a source space :math:`\mathcal{S}` and a target space
:math:`\mathcal{T}`, using parameters :math:`\theta`. For the MNIST digits, for
example we could think of :math:`\mathcal{S} = \mathbb{R}^{28 \times 28} =
\mathbb{R}^{784}` (i.e., the space of all 28×28 images), and for classifying the
MNIST digits we could think of :math:`\mathcal{T} = \mathbb{R}^{10}`.

This mapping is assumed to be fairly complex. If it were not -- if you could
capture the mapping using a simple expression like :math:`F_a(x) = ax^2` -- then
we would just use the expression directly and not need to deal with an entire
network. So if the mapping is complex, we will do a couple of things to make our
problem tractable. First, we will assume some structure for :math:`F_\theta`.
Second, we will fit our model to some set of data that we have obtained, so that
our parameters :math:`\theta` are tuned to the problem at hand.

Graph structure
---------------

.. tikz::
   :libs: arrows

   [thick,->,>=stealth',rectangle,minimum size=10mm,node distance=25mm,rounded corners=3mm]
   \node (dots) at (0, 0) {$\dots$};
   \node[draw] (h1) [left of=dots] {Layer 1} edge (dots);
   \node[draw] (input) [left of=h1] {Input} edge (h1);
   \node[draw] (hkm1) [right of=dots] {Layer $k-1$} edge[<-] (dots);
   \node[draw] (output) [right of=hkm1] {Output} edge[<-] (hkm1);

The mapping :math:`F_\theta` is implemented in neural networks by assuming a
specific, layered form. Computation nodes -- also called units or (sometimes)
neurons -- are arranged in a :math:`k+1` partite graph, with layer :math:`k`
containing :math:`n_k` nodes. The number of input nodes in the graph is referred
to below as :math:`n_0`.

A **weight matrix** :math:`W^k \in \mathbb{R}^{n_{k-1} \times n_k}` specifies
the strength of the connection between nodes in layer :math:`k` and those in
layer :math:`k-1` -- all other pairs of nodes are typically not connected. Each
layer of nodes also has a **bias vector** that determines the offset of each
node from the origin. Together, the parameters :math:`\theta` of the model are
these :math:`k` weight matrices and :math:`k` bias vectors (there are no weights
or biases for the input nodes in the graph).

Local computation
-----------------

.. tikz::
   :libs: arrows

   [thick,->,>=stealth',circle,minimum size=10mm,node distance=10mm,below,near start]
   \node[draw] (z) at (0, 0) {$\sum$};
   \node[draw] (x) at (20mm, 1.5mm) {$z_i^k$} edge[<-] (z);
   \node[draw] (b) at (-30mm, 0) {$z_j^{k-1}$} edge node {$w^k_{ji}$} (z);
   \node (adots) [above of=b] {$\vdots$};
   \node[draw] (a) [above of=adots] {$z_1^{k-1}$} edge node {$w^k_{1i}$} (z);
   \node (cdots) [below of=b] {$\vdots$};
   \node[draw] (c) [below of=cdots] {$z_{n_{k-1}}^{k-1}$} edge node [midway] {$w^k_{n_{k-1}i}$} (z);
   \node[draw] (bias) at (0, -20mm) {$b^k_i$} edge (z);

In a standard feedforward network, each node :math:`i` in layer :math:`k`
receives inputs from all nodes in layer :math:`k-1`, then transforms the
weighted sum of these inputs:

.. math::
   z_i^k = \sigma\left( b_i^k + \sum_{j=1}^{n_{k-1}} w^k_{ji} z_j^{k-1} \right)

where :math:`\sigma: \mathbb{R} \to \mathbb{R}` is an "activation function."
Although many functions will work, typical choices of the activation function
are:

:linear: :math:`\sigma(z) = z`
:rectified linear: :math:`\sigma(z) = \max(0, z)`
:logistic sigmoid: :math:`\sigma(z) = (1 + e^{-z})^{-1}`.

Most activation functions are chosen to incorporate a nonlinearity, since a
model with even multiple linear layers cannot capture nonlinear phenomena. Nodes
in the input layer are assumed to have linear activation (i.e., the input nodes
simply represent the state of the input data), and nodes in the output layer
might have linear or nonlinear activations depending on the modeling task.

Usually all hidden nodes in a network share the same activation function, but
this is not required.

Datasets
--------

Now that we have defined some structure for the model, we will probably need to
fit the parameters of the model to some empirical data. We assume that we have
obtained a set of :math:`M` samples from a distribution :math:`p(\mathcal{S},
\mathcal{T})` over the two spaces that are relevant for our problem.

The data that we need will depend on the task at hand. For a classification task
using the MNIST digits, for example, we will need some samples of
:math:`\mathcal{S} = \mathbb{R}^{28\times 28}` (e.g., pixel arrays of MNIST
digits) as well as the accompanying **labels** from :math:`\mathcal{T} =
\mathbb{R}^{10}` (e.g., vectors with ten entries, each representing the
probability of the corresponding digit class). For an autoencoder or density
estimation task, we only need the **unlabeled** samples from
:math:`\mathcal{S}`.

The samples from :math:`\mathcal{S}` will be referred to below as :math:`x`,
while the samples (labels) from :math:`\mathcal{T}` will be referred to below as
:math:`k`. If many samples are grouped together, we'll assume they are rows in a
matrix :math:`X` or :math:`K`.

.. note::

    In most mathematics treatments, samples are usually treated as column
    vectors. However, in the ``theanets`` library, as well as many other
    Python-based machine learning libraries, these quantities are treated as
    rows. To avoid confusion with the coding world, the math on this page
    assumes row vectors and row-oriented matrices.

A final note about datasets: We will assume that all data have been
mean-centered; that is, we compute the mean for each column (variable) of
:math:`X` and subtract that value from the column. Mean-centering is an easy and
important preprocessing step for almost any dataset, so we assume it implicitly
in everything below.

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

.. _models-autoencoders:

Autoencoders
============

Some types of neural network models have been shown to learn useful features
from a set of data without requiring any label information. This learning task
is often referred to as feature learning or manifold learning. A class of neural
network architectures known as autoencoders are ideally suited for this task. An
autoencoder takes as input a data sample and attempts to produce the same data
sample as its output. Formally, an autoencoder defines a mapping from a source
space to itself:

.. math::
   F_\theta: \mathcal{S} \to \mathcal{S}

Often, this mapping can be decomposed into an "encoding" stage
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
:class:`Autoencoder <theanets.feedforward.Autoencoder>` class::

  exp = theanets.Experiment(theanets.Autoencoder)

The ``layers`` parameter is required to define such a model; it can be provided
on the command-line by using ``--layers A B C ... A``, or in your code::

  exp = theanets.Experiment(
      theanets.Autoencoder,
      layers=(A, B, C, ..., A))

.. note::
   Command-line arguments do not work when running ``theanets`` code in IPython;
   within IPython, all parameters must be specified as keyword arguments.

Finally, a subset of autoencoders with an odd-length, palindromic number of
layers can be defined as having **tied weights** whenever the parameters from
the decoder are the transpose of the parameters from the encoder. Tied-weights
autoencoders form an interesting subset of autoencoder models.

Let's look at a few example models that fall into the autoencoder class.

Single-layer autoencoders
-------------------------

Although the class of autoencoder models is quite large (any :math:`k` partite
graph like the one described above, having the same number of input and output
nodes would count). However, a very interesting class of these models has just
one hidden layer, and uses a linear activation on its output nodes:

.. math::
   F_\theta(x) = \sigma(x W_e + b_e) W_d + b_d

Here, :math:`\sigma` is the activation of the nodes in the hidden layer, and
:math:`W_e`, :math:`W_d`, :math:`b_e`, and :math:`b_d` are the weights and bias
of the "encoding" and "decoding" layers of the network. The trainable parameters
are :math:`\theta = (W_e, W_d, b_e, b_d)`.

To train the weights and biases in the network, an autoencoder typically
optimizes a squared-error reconstruction loss:

.. math::
   J(X, \theta) = \frac{1}{M} \sum_{i=1}^M \left\| \sigma(x_i W_e + b_e) W_d + b_d - x_i \right\|_2^2 + \lambda R(X, \theta)

This optimization process could result in a trivial model, depending on the
setup of the network. In particular, if the number of hidden features
:math:`n_z` is not less than the number of input variables :math:`n_0`, then
with linear hidden activations :math:`\sigma(z) = z`, identity weights
:math:`W_e = W_d = I`, and zero bias :math:`b_e = b_d = 0`, an autoencoder as
defined above implements the identity transform:

.. math::
   F_\theta(x) = x

Even if the hidden unit activations are nonlinear, the network is capable of
learning an identity transform as long as :math:`n_z \ge n_0`. But things get
much more interesting when an autoencoder network is forced to reproduce the
input under some constraint. These constraints can be implemented either through
the structure of the network, or by adding a regularizer. Both of these
approaches will be discussed below.

PCA
```

One way to prevent a model from learning trivial latent representations is to
force the latent space to be smaller than the space where the data live. One of
the most popular techniques for doing this is Principal Component Analysis (PCA)
[Hot33]_. The principal components (PCs) of a dataset are the set of orthogonal
directions :math:`U` (i.e., a rotation) that capture the maximal variance in a
dataset. Each PC :math:`u_i` is scaled by the amount of variance :math:`s_i` in
the corresponding direction of the data, so the first PC captures the most
variance, the second PC the second-most variance, and so forth.

Let's assume we have computed a PCA transform :math:`W = U\diag(S)` for a
dataset :math:`X`. Then we can "encode" the dataset by projecting it into the PC
space using matrix multiplication to rotate and then scale the data:

.. math:: Z = XU\diag(S)

If we wish to "decode" this representation of the data, we can project it back
into the data space by doing another matrix multiplication to un-scale the data
and rotate it back:

.. math::
   \hat{X} = Z\diag(\frac{1}{S})U^\top = X U \diag(S) \diag(1/S) U^\top = X U U^\top

If we have the same number of principal components as variables in our dataset,
then :math:`UU^\top = I` and :math:`\hat{X} = X`. However, if we restrict our PC
representation to a smaller number of dimensions than we have in our data, we
are performing *dimensionality reduction* in a way that is guaranteed to
preserve the most variance in the data. In other words, our transform
:math:`U\diag(S)` minimizes the squared-error loss:

.. math::
   J(X) = \frac{1}{M} \sum_{i=1}^M \left\| \hat{x}_i - x_i \right\|_2^2

.. math::
   J(X) = \frac{1}{M} \sum_{i=1}^M \left\| x_i U U^\top - x_i \right\|_2^2

Given this way of looking at PCA, we can see that it is really a sort of linear
autoencoder with tied weights! To be more precise, optimizing the loss
formulation immediately above is guaranteed to recover the same *subspace* as
the PCA transform, even though the individual features are not necessarily
guaranteed to be the same.

To implement such a model in ``theanets``, we only need to provide the following
hyperparameters::

  pca = theanets.Experiment(
      theanets.Autoencoder,
      tied_weights=True,
      hidden_activation='linear',
      layers=(n_0, n_z, n_0),
  )

This type of model has the additional advantage that it is relatively easy to
train, because the entire model is linear!

In actuality, if your dataset is not too large, it's even easier to use a
closed-form solution to compute the PCA transform; however, looking at PCA in
this way, using a neural network framework, will serve as a good mental bridge
to the sorts of models that will be introduced later on.

ICA
```

For PCA, we had to use an *undercomplete* hidden representation to prevent the
model from learning a trivial identity transform. This is problematic for a
couple of reasons, but from a modeling perspective one of the worst is that the
features computed by PCA are often "tangled together" to represent each of the
points in our dataset. That is, a single PCA feature is often difficult to
interpret by itself; instead, the entire set of PCs is required to yield a
reasonable representation of a data point.

For example, if PCA is performed on a set of image data, the PCs are typically
close to a Fourier basis for the space of images being processed; this
representation does in fact capture the most variance in the data, but any
individual PC only captures one of the spatial frequencies in an image---a
relatively large part of the entire set of PCs must be used to reconstruct an
image with good fidelity.

If instead we wanted to learn an *overcomplete* feature set (i.e., with
:math:`n_z > n_0`), or if we wanted to learn some features of our data that were
not dependent on the others, we could encourage the model to learn a non-trivial
representation of the data by adding a regularizer that specifies how the
features should behave.

One good intuition for introducing a regularizer at this point is to assume that
latent features should be used independently. We can translate that into
mathematics by requiring that the model reproduce the input data using "as
little" feature representation as possible and add an :math:`\ell_1` penalty to
the hidden representation:

.. math::
   J(X, W) = \left\| WW^\top x - x \right\|_2^2 + \lambda \left\| W^\top x \right\|_1

This model, called RICA [Le11]_ ("ICA with a reconstruction cost"), is actually
equivalent to an existing statistical model called Independent Component
Analysis [Jut91]_ [Hyv97]_, which can be trained by maximizing the
non-gaussian-ness (e.g., the kurtosis) of the features. Here, we force the model
to use a sparse representation while still using linear encoding and decoding
with tied weights.

In ``theanets``, we can create such a model by including a sparsity penalty on
the hidden layer::

  rica = theanets.Experiment(
      theanets.Autoencoder,
      tied_weights=True,
      hidden_activation='linear',
      hidden_l1=1,
      layers=(n_0, n_z, n_0),
  )

This model does not have a simple closed-form solution, so an iterative
optimization procedure is just what we need to learn good parameters for the
model.

.. _models-sparse-autoencoder:

Sparse autoencoders
-------------------

RICA models (and ICA generally) are a subset of a more general class of
autoencoder called a *sparse autoencoder* [Glo11]_. Sparse autoencoders
generalize the RICA formulation by adding:

- different encoding and decoding weights,
- bias terms, and
- a nonlinearity at the hidden layer.

Like RICA, however, sparse autoencoders assign a regularization penalty to the
hidden activation of the model:

.. math::
   J(X, \theta) = \frac{1}{M} \sum_{i=1}^M \left\| \sigma(x_i W_e + b_e) W_d +
   b_d - x_i \right\|_2^2 + \lambda\left\| \sigma(x_i W_e + b_e) \right\|_1

The sparsity penalty forces the encoder and decoder of the autoencoder model to
cooperate together to represent the input using as little of the latent space as
possible.

To create a sparse autoencoder in ``theanets``, just use the RICA formulation
but omit the tied weights and linear activation::

  sparse = theanets.Experiment(
      theanets.Autoencoder,
      hidden_l1=1,
      layers=(n_0, n_z, n_0),
  )

Sparse autoencoders can also be created with more than one hidden layer.

.. _models-denoising-autoencoder:

Denoising autoencoders
----------------------

.. _models-regression:

Regression
==========

.. math::
   J(X, K, \theta) = \frac{1}{M} \sum_{i=1}^M \| F_\theta(x_i) - k_i \|_2^2 + R(\theta)

.. _models-classification:

Classification
==============

.. math::
   J(X, K, \theta) = \frac{1}{M} \sum_{i=1}^M \| F_\theta(x_i) - k_i \|_2^2 + R(\theta)

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
   J(\cdot) = \dots + \frac{\lambda}{2} \| \theta \|_2^2

If the loss :math:`J(\cdot)` represents some approximation to the log-posterior
distribution of the model parameters given the data

.. math::
   J(\cdot) = \log p(\theta|x) \propto \dots + \frac{\lambda}{2} \| \theta \|_2^2

then the term with the :math:`L_2` norm on the parameters is like an unscaled
Gaussian distribution.

Sparsity
--------

Sparse models have been shown to capture regularities seen in the mammalian
visual cortex [Ols94]_. In addition, sparse models in machine learning are often
more performant than "dense" models (i.e., models without restriction on the
hidden representation) [Lee08]_. Furthermore, sparse models tend to yield latent
representations that are more interpretable to humans than dense models
[Tib96]_.

References
==========

.. [Glo11] X Glorot, A Bordes, Y Bengio. "Deep sparse rectifier neural
           networks." In *Proc AISTATS*, 2011.

.. [Hot33] H Hotelling. "Analysis of a Complex of Statistical Variables Into
           Principal Components." *Journal of Educational Psychology*
           **24**:417-441 & 498-520, 1933.

.. [Hyv97] A Hyvärinen, "Independent Component Analysis by Minimization of
           Mutual Information." University of Helsinki Tech Report, 1997.

.. [Jut91] C Jutten, J Herault. "Blind separation of sources, part I: An
           adaptive algorithm based on neuromimetic architecture." *Signal
           Processing* **24**:1-10, 1991.

.. [Le11] QV Le, A Karpenko, J Ngiam, AY Ng. "ICA with reconstruction cost for
          efficient overcomplete feature learning." In *Proc NIPS*, 2011.

.. [Lee08] H Lee, C Ekanadham, AY Ng. "Sparse deep belief net model for visual
           area V2." In *Proc. NIPS*, 2008.

.. [Ols94] B Olshausen, DJ Field. "Emergence of simple-cell receptive fields
           properties by learning a sparse code for natural images." *Nature*
           **381** 6583:607-609, 1994.

.. [Sut13] I Sutskever, J Martens, G Dahl, GE Hinton. "On the importance of
           initialization and momentum in deep learning." In *Proc ICML*, 2013.
           http://jmlr.csail.mit.edu/proceedings/papers/v28/sutskever13.pdf

.. [Tib96] R Tibshirani. "Regression shrinkage and selection via the lasso."
           *Journal of the Royal Statistical Society: Series B (Methodological)*
           267-288, 1996.
