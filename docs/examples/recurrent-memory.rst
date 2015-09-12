==========================
Remembering Network Inputs
==========================

Recurrent neural networks are a family of network models whose computation graph
contains a cycle---that is, there are some layers in a recurrent network whose
outputs at a certain time step depend not only on the inputs at that time step,
but also on the state of the network at some previous time step as well.

Recurrent networks, while often quite tricky to train, can be used to solve
difficult modeling tasks. Thanks to recent advances in optimization algorithms,
recurrent networks are enjoying a resurgence in popularity and have been shown
to be quite effective at a number of different temporal modeling tasks.

In this section we consider a classic task for a recurrent network: remembering
data from past inputs. In this task, a network model receives one input value at
each time step. The network is to remember the first :math:`k` values, then wait
for :math:`t` time steps, and then reproduce the first :math:`k` values that it
saw. Effectively the model must ignore the inputs after time step :math:`k` and
start producing the desired output at time step :math:`k + t`.

Defining the model
==================

We'll set up a recurrent model by creating a :class:`recurrent regression
<theanets.recurrent.Regressor>` instance::

  net = theanets.recurrent.Regressor(layers=[1, ('lstm', 10), 1])

Our network has three layers: the first just has one input unit, the next is a
Long Short-Term Memory (LSTM) recurrent layer with ten units, and the output is
a linear layer with just one output unit. This is just one way of specifying
layers in a network; for more details see
:ref:`guide-creating-specifying-layers`.

Training the model
==================

The most difficult part of training this model is creating the required data. To
compute the loss for a recurrent regression model in ``theanets``, we need to
provide two arrays of data---one input array, and one target output array. Each
of these arrays must have three dimensions: the first is time, the second is the
batch size, and the third is the number of inputs/outputs in the dataset.

For the memory task, we can easily create random arrays with the appropriate
shape. We just need to make sure that the last :math:`k` time steps of the
output are set to the first :math:`k` time steps of the input::

  T = 20
  K = 3
  BATCH_SIZE = 32

  def generate():
      s, t = np.random.randn(2, T, BATCH_SIZE, 1).astype('f')
      s[:K] = t[-K:] = np.random.randn(K, BATCH_SIZE, 1)
      return [s, t]

In ``theanets``, data can be provided to a trainer in several ways; here we've
used a callable that generates batches of data for us. See
:ref:`guide-training-providing-data` for more information.

Having set up a way to create training data, we just need to pass this along to
our training algorithm::

  net.train(generate, algo='rmsprop')

This process will adjust the weights in the model so that the outputs of the
model, given the inputs, will be closer and closer to the targets that we
provide.
