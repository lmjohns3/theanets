=======================
Hacking on ``theanets``
=======================

.. _hacking-extending:

Creating New Models
===================

.. _hacking-regularizers:

Defining Custom Regularizers
----------------------------

.. _hacking-costs:

Defining Custom Cost Functions
------------------------------

It's pretty straightforward to create models in ``theanets`` that use cost
functions that are different from the predefined :class:`Classifier
<theanets.feedforward.Classifier>` (which uses binary cross-entropy) and
:class:`Regressor <theanets.feedforward.Regressor>` (which uses mean squared
error). To define by a model with a new cost function, just create a new
subclass and override the ``cost`` property on your subclass. For example, to
create a regression model that uses mean absolute error::

  class MaeRegressor(theanets.Regressor):
      @property
      def cost(self):
          err = self.outputs[-1] - self.targets
          return TT.mean(abs(err).sum(axis=1))

Your cost function must return a theano expression that reflects the cost for
your model.

.. _hacking-contributing:

Contributing
============
