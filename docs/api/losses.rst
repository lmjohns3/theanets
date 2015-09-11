.. _losses:

==============
Loss Functions
==============

.. _losses-weighted:

Using Weighted Targets
======================

By default, the network models available in ``theanets`` treat all inputs as
equal when computing the loss for the model. For example, a regression model
treats an error of 0.1 in component 2 of the output just the same as an error of
0.1 in component 3, and each example of a minibatch is treated with equal
importance when training a classifier.

However, there are times when all inputs to a neural network model are not to be
treated equally. This is especially evident in recurrent models: sometimes, the
inputs to a recurrent network might not contain the same number of time steps,
but because the inputs are presented to the model using a rectangular minibatch
array, all inputs must somehow be made to have the same size. One way to address
this would be to cut off all inputs at the length of the shortest input, but
then the network is not exposed to all input/output pairs during training.

Weighted targets can be used for any model in ``theanets``. For example, an
:class:`autoencoder <theanets.feedforward.Autoencoder>` could use an array of
weights containing zeros and ones to solve a matrix completion task, where the
input array contains some "unknown" values. In such a case, the network is
required to reproduce the known values exactly (so these could be presented to
the model with weight 1), while filling in the unknowns with statistically
reasonable values (which could be presented to the model during training with
weight 0).

As another example, suppose a :class:`classifier
<theanets.feedforward.Classifier>` model is being trained in a binary
classification task where one of the classes---say, class A---is only present
0.1% of the time. In such a case, the network can achieve 99.9% accuracy by
always predicting class B, so during training it might be important to ensure
that errors in predicting A are "amplified" when computing the loss. You could
provide a large weight for training examples in class A to encourage the model
not to miss these examples.

All of these cases are possible to model in ``theanets``; just include
``weighted=True`` when you create your model::

  net = theanets.recurrent.Autoencoder([3, (10, 'rnn'), 3], weighted=True)

When training a weighted model, the training and validation datasets require an
additional component: an array of floating-point values with the same shape as
the expected output of the model. For example, a non-recurrent Classifier model
would require a weight vector with each minibatch, of the same shape as the
labels array, so that the training and validation datasets would each have three
pieces: ``sample``, ``label``, and ``weight``. Each value in the weight array is
used as the weight for the corresponding error when computing the loss.

.. _losses-custom:

Custom Losses
=============

It's pretty straightforward to create models in ``theanets`` that use different
losses from the predefined :class:`Classifier <theanets.feedforward.Classifier>`
and :class:`Autoencoder <theanets.feedforward.Autoencoder>` and
:class:`Regressor <theanets.feedforward.Regressor>` models. (The classifier uses
categorical cross-entropy (XE) as its default loss, and the other two both use
mean squared error, MSE.)

To define a model with a new loss, just create a new :class:`Loss
<theanets.losses.Loss>` subclass and specify its name when you create your
model. For example, to create a regression model that uses a step function
averaged over all of the model inputs::

  class Step(theanets.Loss):
      def __call__(self, outputs):
          return (outputs[self.output_name] > 0).mean()

  net = theanets.Regressor([5, 6, 7], loss='step')

Your loss function implementation must return a Theano expression that reflects
the loss for your model. If you wish to make your loss work with weighted
outputs, you will also need to include a case for having weights::

  class Step(theanets.Loss):
      def __call__(self, outputs):
          step = outputs[self.output_name] > 0
          if self._weights:
              return (self._weights * step).sum() / self._weights.sum()
          else:
              return step.mean()

Source
======

.. automodule:: theanets.losses
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   Loss
   CrossEntropy
   GaussianLogLikelihood
   Hinge
   KullbackLeiblerDivergence
   MaximumMeanDiscrepancy
   MeanAbsoluteError
   MeanSquaredError

