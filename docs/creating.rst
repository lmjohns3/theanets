==================
Creating a Network
==================

.. _creating-predefined-models:

Predefined Models
=================

.. _creating-specifying-layers:

Specifying Layers
=================

One of the most critical bits of creating a neural network model is specifying
how the layers of the network are arranged and connected together. There are
very few limits to the complexity of possible neural network architectures, so
it will never be possible to specify all combinations using a single,
easy-to-use markup. However, ``theanets`` tries to make it easy to create
many common types of layer configurations.

When you create a network model, the ``layers`` keyword argument is used to
specify the layers for your network. This keyword argument must be a sequence
specifying the layers; there are four options for the values in this sequence.

- If a value is an integer, it is interpreted as the size of a vanilla,
  fully-connected feedforward layer. All options for the layer are set to their
  defaults (e.g., the activation for a hidden layer will be given by the
  ``hidden_activation`` configuration parameter, which defaults to a logistic
  sigmoid).
- If a value is a tuple, it must contain an integer and may contain a string.
  The integer in the tuple specifies the size of the layer. If there is a
  string, and the string names a valid layer type (e.g., ``'tied'``, ``'rnn'``,
  etc.), then this type of layer will be created. Otherwise, the string is
  assumed to name an activation function (e.g., ``'logistic'``, ``'relu'``,
  etc.) and a standard feedforward layer will be created with that activation.
- If a value in this sequence is a dictionary, it must contain either a ``size``
  or an ``nout`` key, which specify the number of units in the layer. It can
  additionally contain an ``activation`` key to specify the activation function
  for the layer (see below), and a ``form`` key to specify the type of layer to
  be constructed (e.g., ``'tied'``, ``'rnn'``, etc.). Additional keys in this
  dictionary will be passed as keyword arguments to
  :func:`theanets.layers.build`.
- Finally, if a value is a :class:`Layer <theanets.layers.Layer>` instance, it
  is simply added to the network model as-is.

As an example, ``(10, 20, 3)`` specifies an input layer with 10 units, one
hidden layer with 20 units, and an output layer with 3 units. In this case,
inputs to the network will be of length 10, and outputs will be of length 3.

Activation functions
--------------------

An activation function (sometimes also called a transfer function) specifies how
the output of a layer is computed from the weighted sums of the inputs. By
default, hidden layers in ``theanets`` use a logistic sigmoid activation
function. Output layers in :class:`Regressor <theanets.feedforward.Regressor>`
and :class:`Autoencoder <theanets.feedforward.Autoencoder>` models use linear
activations (i.e., the output is just the weighted sum of the inputs from the
previous layer), and the output layer in :class:`Classifier
<theanets.feedforward.Classifier>` models uses a softmax activation.

To specify a different activation function for a layer, include an activation
key chosen from the table below. As described above, this can be included in
your model specification either using the ``activation`` keyword argument in a
layer dictionary, or by including the key in a tuple with the layer size.

========   ============================  =============================================
Key        Description                   :math:`g(z) =`
========   ============================  =============================================
linear     linear                        :math:`z`
sigmoid    logistic sigmoid              :math:`(1 + e^{-z})^{-1}`
logistic   logistic sigmoid              :math:`(1 + e^{-z})^{-1}`
tanh       hyperbolic tangent            :math:`\tanh(z)`
softplus   smooth relu approximation     :math:`\log(1 + \exp(z))`
softmax    categorical distribution      :math:`e^z / \sum e^z`
relu       rectified linear              :math:`\max(0, z)`
trel       truncated rectified linear    :math:`\max(0, \min(1, z))`
trec       thresholded rectified linear  :math:`z \mbox{ if } z > 1 \mbox{ else } 0`
tlin       thresholded linear            :math:`z \mbox{ if } |z| > 1 \mbox{ else } 0`
rect:max   truncation                    :math:`\min(1, z)`
rect:min   rectification                 :math:`\max(0, z)`
norm:mean  mean-normalization            :math:`z - \bar{z}`
norm:max   max-normalization             :math:`z / \max |z|`
norm:std   variance-normalization        :math:`z / \mathbb{E}[(z-\bar{z})^2]`
========   ============================  =============================================

.. _creating-specifying-regularizers:

Specifying Regularizers
=======================

One heuristic that can prevent parameters from overtraining on small datasets is
based on the observation that "good" parameter values are typically small: large
parameter values often indicate overfitting. One way to encourage a model to use
small parameter values is to assume that the parameter values are sampled from a
posterior distribution over parameters, conditioned on observed data. In this
way of thinking about parameters, we can manipulate the prior distribution of
the parameter values to express our knowledge as modelers of the problem at
hand.

If you want to set up a more sophisticated model like a classifier with sparse
hidden representations, you can add regularization hyperparameters when you
create your experiment::

  exp = theanets.Experiment(
      theanets.Classifier,
      layers=(784, 1000, 784),
      hidden_l1=0.1)

Here we've specified that our model has a single, overcomplete hidden layer, and
the activity of the hidden units in the network will be penalized with a 0.1
coefficient.

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

.. _creating-custom-models:

Defining Custom Models
======================

.. _creating-custom-regularizers:

Defining Custom Regularizers
----------------------------

.. _creating-custom-errors:

Defining Custom Error Functions
-------------------------------

It's pretty straightforward to create models in ``theanets`` that use different
error functions from the predefined :class:`Classifier
<theanets.feedforward.Classifier>` (which uses categorical cross-entropy) and
:class:`Autoencoder <theanets.feedforward.Autoencoder>` and :class:`Regressor
<theanets.feedforward.Regressor>` (which both use mean squared error, MSE). To
define by a model with a new cost function, just create a new :class:`Network
<theanets.feedforward.Network>` subclass and override the ``error`` property.

For example, to create a regression model that uses mean absolute error (MAE)
instead of MSE::

  class MaeRegressor(theanets.Regressor):
      @property
      def error(self):
          return TT.mean(abs(self.outputs[-1] - self.targets))

Your cost function must return a theano expression that reflects the cost for
your model.


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
