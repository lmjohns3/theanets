.. _activations:

====================
Activation Functions
====================

An activation function (sometimes also called a transfer function) specifies how
the final output of a layer is computed from the weighted sums of the inputs.

By default, hidden layers in ``theanets`` use a rectified linear activation
function: :math:`g(z) = \max(0, z)`.

Output layers in :class:`theanets.Regressor <theanets.feedforward.Regressor>`
and :class:`theanets.Autoencoder <theanets.feedforward.Autoencoder>` models use
linear activations (i.e., the output is just the weighted sum of the inputs from
the previous layer: :math:`g(z) = z`), and the output layer in
:class:`theanets.Classifier <theanets.feedforward.Classifier>` models uses a
softmax activation: :math:`g(z) = \exp(z) / \sum\exp(z)`.

To specify a different activation function for a layer, include an activation
key chosen from the table below, or :ref:`create a custom activation
<activations-custom>`. As described in :ref:`guide-creating-specifying-layers`,
the activation key can be included in your model specification either using the
``activation`` keyword argument in a layer dictionary, or by including the key
in a tuple with the layer size:

.. code:: python

  net = theanets.Regressor([10, (10, 'tanh'), 10])

The activations that ``theanets`` provides are:

=========    ============================  ===============================================
Key          Description                   :math:`g(z) =`
=========    ============================  ===============================================
linear       linear                        :math:`z`
sigmoid      logistic sigmoid              :math:`(1 + \exp(-z))^{-1}`
logistic     logistic sigmoid              :math:`(1 + \exp(-z))^{-1}`
tanh         hyperbolic tangent            :math:`\tanh(z)`
softplus     smooth relu approximation     :math:`\log(1 + \exp(z))`
softmax      categorical distribution      :math:`\exp(z) / \sum\exp(z)`
relu         rectified linear              :math:`\max(0, z)`
rect:min     truncation                    :math:`\min(1, z)`
rect:max     rectification                 :math:`\max(0, z)`
rect:minmax  truncation and rectification  :math:`\max(0, \min(1, z))`
norm:mean    mean-normalization            :math:`z - \bar{z}`
norm:max     max-normalization             :math:`z / \max |z|`
norm:std     variance-normalization        :math:`z / \mathbb{E}[(z-\bar{z})^2]`
norm:z       z-score normalization         :math:`(z-\bar{z}) / \mathbb{E}[(z-\bar{z})^2]`
prelu_       relu with parametric leak     :math:`\max(0, z) - \max(0, -rz)`
lgrelu_      relu with leak and gain       :math:`\max(0, gz) - \max(0, -rz)`
maxout_      piecewise linear              :math:`\max_i m_i z`
=========    ============================  ===============================================

.. _prelu: generated/theanets.activations.Prelu.html
.. _lgrelu: generated/theanets.activations.LGrelu.html
.. _maxout: generated/theanets.activations.Maxout.html

Composition
===========

Activation functions can also be composed by concatenating multiple function
names togather using a ``+``. For example, to create a layer that uses a
batch-normalized hyperbolic tangent activation:

.. code:: python

  net = theanets.Regressor([10, (10, 'tanh+norm:z'), 10])

Just like function composition, the order of the components matters! Unlike the
notation for mathematical function composition, the functions will be applied
from left-to-right.

.. _activations-custom:

Custom Activations
==================

To define a custom activation, create a subclass of :class:`theanets.Activation
<theanets.activations.Activation>`, and implement the ``__call__`` method to
make the class instance callable. The callable will be given one argument, the
array of layer outputs to activate.

.. code:: python

  class ThresholdedLinear(theanets.Activation):
      def __call__(self, x):
          return x * (x > 1)

This example activation returns 0 if a layer output is less than 1, or the
output value itself otherwise. In effect it is a linear activation for "large"
outputs (i.e., greater than 1) and zero otherwise. To use it in a model, give
the name of the activation:

.. code:: python

  net = theanets.Regressor([10, (10, 'thresholdedlinear'), 10])
