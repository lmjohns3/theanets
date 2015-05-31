=============
Using a Model
=============

Once you've trained a model, you will probably want to do something useful with
it. If you are working in a production environment, you might want to use the
model to make predictions about incoming data; if you are doing research, you
might want to examine the parameters that the model has learned.

Predicting New Data
===================

For most neural network models, you can compute the "natural" output of the model
layer by calling :func:`Network.predict() <theanets.graph.Network.predict>`::

  results = exp.network.predict(new_dataset)

For :class:`regression <theanets.feedforward.Regressor>` and
:class:`autoencoding <theanets.feedforward.Autoencoder>` models, this method
returns the output of the network when passed the given input dataset. For
:class:`classification <theanets.feedforward.Classifier>` models, this method
returns the predicted classification of the inputs. (To get the actual output of
the network---the posterior class probabilities---for a classifier model, use
:func:`predict_proba() <theanets.feedforward.Classifier.predict_proba>`.)

Regardless of the model, you pass to ``predict()`` a ``numpy`` array containing
data examples along the rows, and the method returns an array containing one row
of output predictions for each row of input data.

You can also compute the activations of all layer outputs in the network using
the :func:`Network.feed_forward() <theanets.feedforward.Network.feed_forward>`
method::

  for name, value in exp.network.feed_forward(new_dataset).items():
      print(abs(layer).sum(axis=1))

This method returns a dictionary that maps layer output names to their
corresponding values for the given input. Like ``predict()``, each output array
contains one row for every row of input data.

Getting Learned Parameters
==========================

The parameters in each layer of the model are available using
:func:`Network.find() <theanets.feedforward.Network.find>`. This method takes
two query terms---either integer index values or string names---and returns a
theano shared variable for the given parameter. The first query term finds a
layer in the network, and the second finds a parameter within that layer.

The ``find()`` method returns a `Theano shared variable`_. To get a numpy array
of the current values of the variable, call ``get_value()`` on the result from
``find()``, like so::

  values = network.find(1, 0).get_value()

For "encoding" layers in the network, this value array contains a feature vector
in each column, and for "decoding" layers (i.e., layers connected to the output
of an autoencoder), the features are in each row.

.. _Theano shared variable: http://deeplearning.net/software/theano/library/compile/shared.html

Visualizing Weights
===================

Many times it is useful to create a plot of the features that the model learns;
this can be useful for debugging model performance, but also for interpreting
the dataset through the "lens" of the learned features.

For example, if you have a model that takes as input a 28×28 MNIST digit, then
you could plot the weight vectors attached to each unit in the first hidden
layer of the model to see what sorts of features the hidden unit detects::

  img = np.zeros((28 * 10, 28 * 10), dtype='f')
  for i, pix in enumerate(exp.network.find(1, 0).get_value().T):
      r, c = divmod(i, 10)
      img[r * 28:(r+1) * 28, c * 28:(c+1) * 28] = pix.reshape((28, 28))
  plt.imshow(img, cmap=plt.cm.gray)
  plt.show()

Here we've taken the weights from the first hidden layer of the model
(``exp.network.find(1, 0)``) and plotted them as though they were 28×28
grayscale images. This is a useful technique for processing images (and, to some
extent, other types of data) because visually inspecting features can give you a
quick sense of how the model interprets its input. In addition, this can serve
as a sanity check---if the features in the model look like TV snow, for example,
the model probably hasn't adapted its weights properly, so something might be
wrong with the training process.
