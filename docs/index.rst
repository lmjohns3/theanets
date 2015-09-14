========
THEANETS
========

The ``theanets`` package is a deep learning and neural network toolkit. It is
written in Python to interoperate with excellent tools like ``numpy`` and
``scikit-learn``, and it uses Theano_ to accelerate computations when possible
using your GPU. The package aims to provide:

- a simple API for building and training common types of neural network models;
- thorough documentation;
- easy-to-read code;
- and, under the hood, a fully expressive graph computation framework.

The package strives to "make the easy things easy and the difficult things
possible." Please try it out, and let us know what you think!

The source code for ``theanets`` lives at http://github.com/lmjohns3/theanets,
the documentation lives at http://theanets.readthedocs.org, and announcements
and discussion happen on the `mailing list`_.

.. _Theano: http://deeplearning.net/software/theano/
.. _mailing list: https://groups.google.com/forum/#!forum/theanets

Quick Start: Classification
===========================

Suppose you want to create a classifier and train it on some 100-dimensional
data points that you've classified into 10 categories. No problem! With just a
few lines you can (a) provide some data, (b) build and (c) train a model,
and (d) evaluate the model::

  import climate
  import theanets
  from sklearn.datasets import make_classification
  from sklearn.metrics import confusion_matrix

  climate.enable_default_logging()

  # Create a classification dataset.
  X, y = make_classification(
      n_samples=3000, n_features=100, n_classes=10, n_informative=10)
  X = X.astype('f')
  y = y.astype('i')
  cut = int(len(X) * 0.8)  # training / validation split
  train = X[:cut], y[:cut]
  valid = X[cut:], y[cut:]

  # Build a classifier model with 100 inputs and 10 outputs.
  net = theanets.Classifier([100, 10])

  # Train the model using SGD with momentum.
  net.train(train, valid, algo='sgd', learning_rate=1e-4, momentum=0.9)

  # Show confusion matrices on the training/validation splits.
  for label, (X, y) in (('training:', train), ('validation:', valid)):
      print(label)
      print(confusion_matrix(y, net.predict(X)))

Layers
------

The model above is quite simplistic! Make it a bit more sophisticated by adding
a hidden layer::

  net = theanets.Classifier([100, 1000, 10])

In fact, you can just as easily create 3 (or any number of) hidden layers::

  net = theanets.Classifier([
      100, 1000, 1000, 1000, 10])

By default, hidden layers use the relu transfer function. By passing a tuple
instead of just an integer, you can change some of these layers to use different
:mod:`activations <theanets.activations>`::

  maxout = (1000, 'maxout:4')  # maxout with 4 pieces.
  net = theanets.Classifier([
      100, 1000, maxout, (1000, 'tanh'), 10])

By passing a dictionary instead, you can specify even more attributes of each
:mod:`layer <theanets.layers.base>`, like how its parameters are initialized::

  # Sparsely-initialized layer with large nonzero weights.
  foo = dict(name='foo', size=1000, std=1, sparsity=0.9)
  net = theanets.Classifier([
      100, foo, (1000, 'maxout:4'), (1000, 'tanh'), 10])

Specifying layers is the heart of building models in ``theanets``. Read more
about this in :ref:`guide-creating-specifying-layers`.

Regularization
--------------

Adding regularizers is easy, too! Just pass them to the training method. For
instance, you can train up a sparse classification model with weight decay::

  # Penalize hidden-unit activity (L1 norm) and weights (L2 norm).
  net.train(train, valid, hidden_l1=0.001, weight_l2=0.001)

In ``theanets`` dropout is treated as a regularizer and can be set on many
layers at once::

  net.train(train, valid, hidden_dropout=0.5)

or just on a specific layer::

  net.train(train, valid, dropout={'foo:out': 0.5})

Similarly, you can add Gaussian noise to any of the layers (here, just to the
input layer)::

  net.train(train, valid, input_noise=0.3)

Optimization Algorithms
-----------------------

You can optimize your model using any of the algorithms provided by downhill_
(SGD, NAG, RMSProp, ADADELTA, etc.), or additionally using a couple of
:mod:`pretraining methods <theanets.trainer>` specific to neural networks.

.. _downhill: http://downhill.readthedocs.org/
.. _pretraining methods: http://theanets.readthedocs.org/en/latest/reference.html#module-theanets.trainer

You can also make as many successive calls to :func:`train()
<theanets.graph.Network.train>` as you like. Each call can include different
training algorithms::

  net.train(train, valid, algo='rmsprop')
  net.train(train, valid, algo='nag')

different learning hyperparameters::

  net.train(train, valid, algo='rmsprop', learning_rate=0.1)
  net.train(train, valid, algo='rmsprop', learning_rate=0.01)

and different regularization hyperparameters::

  net.train(train, valid, input_noise=0.7)
  net.train(train, valid, input_noise=0.3)

Training models is a bit more art than science, but ``theanets`` tries to make
it easy to evaluate different training approaches. Read more about this in
:ref:`guide-training`.

Quick Start: Recurrent Models
=============================

Recurrent neural networks are becoming quite important for many sequence-based
tasks in machine learning; one popular toy example for recurrent models is to
generate text that's similar to some body of training text.

In these models, a recurrent classifier is set up to predict the identity of the
next character in a sequence of text, given all of the preceding characters. The
inputs to the model are the one-hot encodings of a sequence of characters from
the text, and the corresponding outputs are the class labels of the subsequent
character. The ``theanets`` code has a :class:`Text <theanets.recurrent.Text>`
helper class that provides easy encoding and decoding of text to and from
integer classes; using the helper makes the top-level code look like::

  import numpy as np, re, theanets

  chars = re.sub(r'\s+', ' ', open('corpus.txt').read().lower())
  txt = theanets.recurrent.Text(chars, min_count=10)
  A = 1 + len(txt.alpha)  # of letter classes

  # create a model to train: input -> gru -> relu -> softmax.
  net = theanets.recurrent.Classifier([
      A, (100, 'gru'), (1000, 'relu'), A])

  # train the model iteratively; draw a sample after every epoch.
  seed = txt.encode(txt.text[300017:300050])
  for tm, _ in net.itertrain(txt.classifier_batches(100, 32), momentum=0.9):
      print('{}|{} ({:.1f}%)'.format(
          txt.decode(seed),
          txt.decode(net.predict_sequence(seed, 40)),
          100 * tm['acc']))

This example uses several features of ``theanets`` that make modeling neural
networks fun and interesting. The model uses a layer of :class:`Gated Recurrent
Units <theanets.layers.recurrent.GRU>` to capture the temporal dependencies in
the data. It also `uses a callable`_ to provide data to the model, and takes
advantage of `iterative training`_ to sample an output from the model after each
training epoch.

.. _uses a callable: http://downhill.readthedocs.org/en/stable/guide.html#data-using-callables
.. _iterative training: http://downhill.readthedocs.org/en/stable/guide.html#iterative-optimization

To run this example, download a text you'd like to model (e.g., Herman
Melville's *Moby Dick*) and save it in ``corpus.txt``::

  curl http://www.gutenberg.org/cache/epub/2701/pg2701.txt > corpus.txt

Then when you run the script, the output might look something like this
(abbreviated to show patterns)::

  used for light, but only as an oi|pr vgti ki nliiariiets-a, o t.;to niy  , (16.6%)
  used for light, but only as an oi|s bafsvim-te i"eg nadg tiaraiatlrekls tv (20.2%)
  used for light, but only as an oi|vetr uob bsyeatit is-ad. agtat girirole, (28.5%)
  used for light, but only as an oi|siy thinle wonl'th, in the begme sr"hey  (29.9%)
  used for light, but only as an oi|nr. bonthe the tuout honils ohe thib th  (30.5%)
  used for light, but only as an oi|kg that mand sons an, of,rtopit bale thu (31.0%)
  used for light, but only as an oi|nsm blasc yan, ang theate thor wille han (32.1%)
  used for light, but only as an oi|b thea mevind, int amat ars sif istuad p (33.3%)
  used for light, but only as an oi|msenge bie therale hing, aik asmeatked s (34.1%)
  used for light, but only as an oi|ge," rrermondy ghe e comasnig that urle  (35.5%)
  used for light, but only as an oi|s or thartich comase surt thant seaiceng (36.1%)
  used for light, but only as an oi|s lot fircennor, unding dald bots trre i (37.1%)
  used for light, but only as an oi|st onderass noptand. "peles, suiondes is (38.2%)
  used for light, but only as an oi|gnith. s. lited, anca! stobbease so las, (39.3%)
  used for light, but only as an oi|chics fleet dong berieribus armor has or (40.1%)
  used for light, but only as an oi|cs and quirbout detom tis glome dold pco (41.1%)
  used for light, but only as an oi|nht shome wand, the your at movernife lo (42.0%)
  used for light, but only as an oi|r a reald hind the, with of the from sti (43.0%)
  used for light, but only as an oi|t beftect. how shapellatgen the fortower (44.0%)
  used for light, but only as an oi|rtucated fanns dountetter from fom to wi (45.2%)
  used for light, but only as an oi|r the sea priised tay queequings hearhou (46.8%)
  used for light, but only as an oi|ld, wode, i long ben! but the gentived.  (48.0%)
  used for light, but only as an oi|r wide-no nate was him. "a king to had o (49.1%)
  used for light, but only as an oi|l erol min't defositanable paring our. 4 (50.0%)
  used for light, but only as an oi|l the motion ahab, too, and relay in aha (51.0%)
  used for light, but only as an oi|n dago, and contantly used the coil; but (52.3%)
  used for light, but only as an oi|l starbuckably happoss of the fullies ti (52.4%)
  used for light, but only as an oi|led-bubble most disinuan into the mate-- (53.3%)
  used for light, but only as an oi|len. ye?' 'tis though moby starbuck, and (53.6%)
  used for light, but only as an oi|l, and the pequodeers. but was all this: (53.9%)
  used for light, but only as an oi|ling his first repore to the pequod, sym (54.4%)
  used for light, but only as an oi|led escried; we they like potants--old s (54.3%)
  used for light, but only as an oi|l-ginqueg! i save started her supplain h (54.3%)
  used for light, but only as an oi|l is, the captain all this mildly bounde (54.9%)

Here, the seed text is shown left of the pipe character, and the randomly
sampled sequence follows. In parantheses are the per-character accuracy values
on the training set while training the model. The pattern of learning proceeds
from almost-random character generation, to producing groups of letters
separated by spaces, to generating words that seem like they might belong in
*Moby Dick*, things like "captain," "ahab, too," and "constantly used the coil."

Much amusement can be derived from a temporal model extending itself forward in
this way. After all, how else would we ever think of "Pequodeers,"
"Starbuckably," or "Ginqueg"?!

User Guide
==========

.. toctree::
   :maxdepth: 2

   guide

Examples
========

.. toctree::
   :maxdepth: 2
   :glob:

   examples/*

API Documentation
=================

.. toctree::
   :maxdepth: 2
   :glob:

   api/models
   api/layers
   api/activations
   api/losses
   api/regularizers
   api/trainers
   api/utils

.. toctree::
   :hidden:

   api/reference

Indices and tables
==================

- :ref:`genindex`
- :ref:`modindex`
