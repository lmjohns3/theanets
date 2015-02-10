# -*- coding: utf-8 -*-

r'''This module contains a number of classes for modeling neural nets in Theano.

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

.. image:: _static/feedforward_layers.svg

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

.. image:: _static/feedforward_neuron.svg

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
'''

import climate
import gzip
import numpy as np
import pickle
import theano
import theano.tensor as TT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

logging = climate.get_logger(__name__)

from . import layers

FLOAT = theano.config.floatX


def load(filename, **kwargs):
    '''Load an entire network from a pickle file on disk.

    If this function is called without extra keyword arguments, a new network
    will be created using the keyword arguments that were originally used to
    create the pickled network. If this helper function is called with extra
    keyword arguments, they will override arguments that were originally used to
    create the pickled network. This override allows one to, for example, load a
    network that was created with one activation function, and apply a different
    activation function to the existing weights. Some options will cause errors
    if overridden, such as `layers` or `tied_weights`, since they change the
    number of parameters in the model.

    Parameters
    ----------
    filename : str
        Load the keyword arguments and parameters of a network from a pickle
        file at the named path. If this name ends in ".gz" then the input will
        automatically be gunzipped; otherwise the input will be treated as a
        "raw" pickle.

    Returns
    -------
    network : :class:`Network`
        A newly-constructed network, with topology and parameters loaded from
        the given pickle file.
    '''
    opener = gzip.open if filename.lower().endswith('.gz') else open
    handle = opener(filename, 'rb')
    pkl = pickle.load(handle)
    handle.close()
    kw = pkl['kwargs']
    kw.update(kwargs)
    net = pkl['klass'](**kw)
    net.load_params(filename)
    return net


class Network(object):
    '''The network class encapsulates a fully-connected feedforward net.

    In addition to defining standard functionality for feedforward nets, there
    are also many options for specifying topology and regularization, several of
    which must be provided to the constructor at initialization time.

    Parameters
    ----------
    layers : sequence of int, tuple, dict, or :class:`Layer <layers.Layer>`
        A sequence of values specifying the layer configuration for the network.
        For more information, please see :ref:`creating-specifying-layers`.
    hidden_activation : str, optional
        The name of an activation function to use on hidden network layers by
        default. Defaults to 'logistic'.
    output_activation : str, optional
        The name of an activation function to use on the output layer by
        default. Defaults to 'linear'.
    rng : theano RandomStreams object, optional
        Use a specific Theano random number generator. A new one will be created
        if this is None.
    input_noise : float, optional
        Standard deviation of desired noise to inject into input.
    hidden_noise : float, optional
        Standard deviation of desired noise to inject into hidden unit
        activation output.
    input_dropouts : float in [0, 1], optional
        Proportion of input units to randomly set to 0.
    hidden_dropouts : float in [0, 1], optional
        Proportion of hidden unit activations to randomly set to 0.
    decode_from : positive int, optional
        Any of the hidden layers can be tapped at the output. Just specify a
        value greater than 1 to tap the last N hidden layers. The default is 1,
        which decodes from just the last layer.

    Attributes
    ----------
    layers : list of :class:`Layer <layers.Layer>`
        A list of the layers in this network model.
    kwargs : dict
        A dictionary containing the keyword arguments used to construct the
        network.
    '''

    def __init__(self, **kwargs):
        self.layers = []
        self.kwargs = kwargs
        self.inputs = list(self.setup_vars())
        self.setup_layers()

    def setup_vars(self):
        '''Setup Theano variables required by our network.

        The default variable for a network is simply `x`, which represents the
        input to the network.

        Subclasses may override this method to specify additional variables. For
        example, a supervised model might specify an additional variable that
        represents the target output for a particular input.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        # x is a proxy for our network's input, and y for its output.
        self.x = TT.matrix('x')
        return [self.x]

    def setup_layers(self):
        '''Set up a computation graph for our network.

        The default implementation constructs a series of feedforward
        layers---called the "encoder" layers---and then calls
        :func:`setup_decoder` to construct the decoding apparatus in the
        network.

        Subclasses may override this method to construct alternative network
        topologies.
        '''
        if 'layers' not in self.kwargs:
            return

        specs = list(self.encoding_layers)
        rng = self.kwargs.get('rng') or RandomStreams()

        # setup input layer.
        self.layers.append(layers.build('input', specs.pop(0),
            rng=rng,
            name='in',
            dropout=self.kwargs.get('input_dropouts', 0),
            noise=self.kwargs.get('input_noise', 0)))

        # setup "encoder" layers.
        for i, spec in enumerate(specs):
            # if spec is a Layer instance, just add it and move on.
            if isinstance(spec, layers.Layer):
                self.layers.append(spec)
                continue

            # here we set up some defaults for constructing a new layer.
            form = 'feedforward'
            kwargs = dict(
                nin=self.layers[-1].nout,
                rng=rng,
                name='hid{}'.format(len(self.layers)),
                noise=self.kwargs.get('hidden_noise', 0),
                dropout=self.kwargs.get('hidden_dropouts', 0),
                batch_size=self.kwargs.get('batch_size', 64),
                activation=self.kwargs.get('hidden_activation', 'logistic'),
            )

            # by default, spec is assumed to be a lowly integer, giving the
            # number of units in the layer.
            if isinstance(spec, int):
                kwargs['nout'] = spec

            # if spec is a tuple, assume that it contains one or more of the following:
            # - the type of layer to construct (layers.Layer subclass)
            # - the name of a class for the layer (str; if layes.Layer subclass)
            # - the name of an activation function (str; otherwise)
            # - the number of units in the layer (int)
            if isinstance(spec, (tuple, list)):
                for el in spec:
                    try:
                        if issubclass(el, layers.Layer):
                            form = el.__name__
                    except TypeError:
                        pass
                    if isinstance(el, str):
                        if el.lower() in layers.Layer._registry:
                            form = el
                        else:
                            kwargs['activation'] = el
                    if isinstance(el, int):
                        kwargs['nout'] = el
                kwargs['name'] = '{}{}'.format(form, len(self.layers))

            # if spec is a dictionary, try to extract a form and size for the
            # layer, and override our default keyword arguments with the rest.
            if isinstance(spec, dict):
                if 'form' in spec:
                    form = spec['form'].lower()
                    kwargs['name'] = '{}{}'.format(form, len(self.layers))
                if 'size' in spec:
                    kwargs['nout'] = spec['size']
                kwargs.update(spec)

            if isinstance(form, str) and form.lower() == 'bidirectional':
                kwargs['name'] = 'bd{}{}'.format(
                    kwargs.get('worker', 'rnn'), len(self.layers))

            self.layers.append(layers.build(form, **kwargs))

        # setup output layer.
        self.setup_decoder()

    def setup_decoder(self):
        '''Set up the "decoding" computations from layer activations to output.

        The default decoder constructs a single weight matrix for each of the
        hidden layers in the network that should be used for decoding (see the
        `decode_from` parameter) and outputs the sum of the decoders.

        This method can be overridden by subclasses to implement alternative
        decoding strategies.

        Parameters
        ----------
        decode_from : int, optional
            Compute the activation of the output vector using the activations of
            the last N hidden layers in the network. Defaults to 1, which
            results in a traditional setup that decodes only from the
            penultimate layer in the network.
        '''
        sizes = [l.nout for l in self.layers]
        back = self.kwargs.get('decode_from', 1)
        self.layers.append(layers.build(
            'feedforward',
            name='out',
            nin=sizes[-1] if back <= 1 else sizes[-back:],
            nout=self.kwargs['layers'][-1],
            activation=self.output_activation))

    @property
    def output_activation(self):
        return self.kwargs.get('output_activation', 'linear')

    @property
    def encoding_layers(self):
        '''Determine the layers that will be part of the network encoder.

        This method is used by the default implementation of
        :func:`setup_layers` to determine which layers in the network will be
        treated as "encoding" layers. The default is to treat all but the last
        layer as encoders.

        Returns
        -------
        layers : list of int
            A list of integers specifying sizes of the encoder network layers.
        '''
        return self.kwargs['layers'][:-1]

    def _connect(self):
        '''Connect the layers in this network to form a computation graph.

        Returns
        -------
        outputs : list of theano variables
            A list of expressions giving the output of each layer in the graph.
        monitors : list of (name, expression) tuples
            A list of expressions to use when monitoring the network.
        updates : list of update tuples
            A list of updates that should be performed by a theano function that
            computes something using this graph.
        '''
        if not hasattr(self, '_graph'):
            outputs = []
            monitors = []
            updates = []
            for i, layer in enumerate(self.layers):
                if i == 0:
                    # input to first layer is data.
                    inputs = self.x
                elif i == len(self.layers) - 1:
                    # inputs to last layer is output of layers to decode.
                    inputs = outputs[-self.kwargs.get('decode_from', 1):]
                else:
                    # inputs to other layers are outputs of previous layer.
                    inputs = outputs[-1]
                out, mon, upd = layer.output(inputs)
                outputs.append(out)
                monitors.extend(mon)
                updates.extend(upd)
            self._graph = outputs, monitors, updates
        return self._graph

    @property
    def outputs(self):
        return self._connect()[0]

    @property
    def _monitors(self):
        return self._connect()[1]

    @property
    def updates(self):
        return self._connect()[2]

    @property
    def monitors(self):
        '''A sequence of name-value pairs for monitoring the network.

        Names in this sequence are strings, and values are theano variables
        describing how to compute the relevant quantity.

        These monitor expressions are used by network trainers to compute
        quantities of interest during training. The default set of monitors
        consists of:

        - err: the unregularized error of the network
        - X<0.1: percent of units in layer X such that :math:`|a_i| < 0.1`
        - X<0.9: percent of units in layer X such that :math:`|a_i| < 0.9`
        '''
        yield 'err', self.error
        for name, value in self._monitors:
            yield name, value

    @property
    def params(self):
        '''Get a list of the learnable theano parameters for this network.

        This attribute is mostly used by :class:`Trainer
        <theanets.trainer.Trainer>` implementations to compute the set of
        parameters that are tunable in a network.

        Returns
        -------
        params : list of theano variables
            A list of parameters that can be learned in this model.

        '''
        return [p for l in self.layers for p in l.params]

    def find(self, layer, param):
        '''Get a parameter from a layer in the network.

        Parameters
        ----------
        layer : int or str
            The layer that owns the parameter to return.

            If this is an integer, then 0 refers to the input layer, 1 refers
            to the first hidden layer, 2 to the second, and so on.

            If this is a string, the layer with the corresponding name, if any,
            will be used.

        param : int or str
            Name of the parameter to retrieve from the specified layer, or its
            index in the parameter list of the layer.

        Raises
        ------
        KeyError
            If there is no such layer, or if there is no such parameter in the
            specified layer.

        Returns
        -------
        param : theano shared variable
            A shared parameter variable from the indicated layer.
        '''
        for i, l in enumerate(self.layers):
            if layer == i or layer == l.name:
                return l.find(param)
        raise KeyError(layer)

    def feed_forward(self, x):
        '''Compute a forward pass of all layers from the given input.

        Parameters
        ----------
        x : ndarray (num-examples, num-variables)
            An array containing data to be fed into the network. Multiple
            examples are arranged as rows in this array, with columns containing
            the variables for each example.

        Returns
        -------
        layers : list of ndarray (num-examples, num-units)
            The activation values of each layer in the the network when given
            input `x`. For each of the hidden layers, an array is returned
            containing one row per input example; the columns of each array
            correspond to units in the respective layer. The "output" of the
            network is the last element of this list.
        '''
        if not hasattr(self, '_compute'):
            outputs, _, updates = self._connect()
            self._compute = theano.function([self.x], outputs, updates=updates)
        return self._compute(x)

    def predict(self, x):
        '''Compute a forward pass of the inputs, returning the network output.

        Parameters
        ----------
        x : ndarray (num-examples, num-variables)
            An array containing data to be fed into the network. Multiple
            examples are arranged as rows in this array, with columns containing
            the variables for each example.

        Returns
        -------
        y : ndarray (num-examples, num-variables
            Returns the values of the network output units when given input `x`.
            Rows in this array correspond to examples, and columns to output
            variables.
        '''
        return self.feed_forward(x)[-1]

    __call__ = predict

    def save(self, filename):
        '''Save the state of this network to a pickle file on disk.

        Parameters
        ----------
        filename : str
            Save the parameters of this network to a pickle file at the named
            path. If this name ends in ".gz" then the output will automatically
            be gzipped; otherwise the output will be a "raw" pickle.
        '''
        state = dict(klass=self.__class__, kwargs=self.kwargs)
        for layer in self.layers:
            key = '{}-values'.format(layer.name)
            state[key] = [p.get_value() for p in layer.params]
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'wb')
        pickle.dump(state, handle, -1)
        handle.close()
        logging.info('%s: saved model parameters', filename)

    def load_params(self, filename):
        '''Load the parameters for this network from disk.

        Parameters
        ----------
        filename : str
            Load the parameters of this network from a pickle file at the named
            path. If this name ends in ".gz" then the input will automatically
            be gunzipped; otherwise the input will be treated as a "raw" pickle.
        '''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'rb')
        saved = pickle.load(handle)
        handle.close()
        for layer in self.layers:
            for p, v in zip(layer.params, saved['{}-values'.format(layer.name)]):
                p.set_value(v)
        logging.info('%s: loaded model parameters', filename)

    def loss(self, **kwargs):
        '''Return a variable representing the loss for this network.

        The loss includes both the error for the network as well as any
        regularizers that are in place.

        Parameters
        ----------
        weight_l1 : float, optional
            Regularize the L1 norm of unit connection weights by this constant.
        weight_l2 : float, optional
            Regularize the L2 norm of unit connection weights by this constant.
        hidden_l1 : float, optional
            Regularize the L1 norm of hidden unit activations by this constant.
        hidden_l2 : float, optional
            Regularize the L2 norm of hidden unit activations by this constant.
        contractive : float, optional
            Regularize model using the Frobenius norm of the hidden Jacobian.

        Returns
        -------
        loss : theano variable
            A variable representing the loss of this network.
        '''
        hiddens = self.outputs[1:-1]
        regularizers = dict(
            weight_l1=(abs(w).sum() for l in self.layers for w in l.params),
            weight_l2=((w * w).sum() for l in self.layers for w in l.params),
            hidden_l1=(abs(h).mean(axis=0).sum() for h in hiddens),
            hidden_l2=((h * h).mean(axis=0).sum() for h in hiddens),
            contractive=(TT.sqr(TT.grad(h.mean(axis=0).sum(), self.x)).sum()
                         for h in hiddens),
        )
        return self.error + sum(TT.cast(kwargs[weight], FLOAT) * sum(expr)
                                for weight, expr in regularizers.items()
                                if kwargs.get(weight, 0) > 0)


class Autoencoder(Network):
    r'''An autoencoder attempts to reproduce its input.

    Some types of neural network models have been shown to learn useful features
    from a set of data without requiring any label information. This learning
    task is often referred to as feature learning or manifold learning. A class
    of neural network architectures known as autoencoders are ideally suited for
    this task. An autoencoder takes as input a data sample and attempts to
    produce the same data sample as its output. Formally, an autoencoder defines
    a mapping from a source space to itself:

    .. math::
       F_\theta: \mathcal{S} \to \mathcal{S}

    Often, this mapping can be decomposed into an "encoding" stage
    :math:`f_\alpha(\cdot)` and a corresponding "decoding" stage
    :math:`g_\beta(\cdot)` to and from some latent space :math:`\mathcal{Z} =
    \mathbb{R}^{n_z}`:

    .. math::
       \begin{eqnarray*}
       f_\alpha &:& \mathcal{S} \to \mathcal{Z} \\
       g_\beta &:& \mathcal{Z} \to \mathcal{S}
       \end{eqnarray*}

    Autoencoders form an interesting class of models for several reasons. They:

    - require only "unlabeled" data (which is typically easy to obtain),
    - are generalizations of many popular density estimation techniques, and
    - can be used to model the "manifold" or density of a dataset.

    If we have a labeled dataset containing :math:`m` :math:`d`-dimensional
    input samples :math:`X \in \mathbb{R}^{m \times d}`, then the loss that the
    autoencoder model optimizes with respect to the model parameters
    :math:`\theta` is:

    .. math::
       \begin{eqnarray*}
       \mathcal{L}(X, \theta) &=& \frac{1}{m} \sum_{i=1}^m \| F_\theta(x_i) - x_i \|_2^2 + R(X, \theta) \\
       &=& \frac{1}{m} \sum_{i=1}^m \| g_\beta(f_\alpha(x_i)) - x_i \|_2^2 + R(X, \alpha, \beta)
       \end{eqnarray*}

    where :math:`R` is a regularization function.

    A generic autoencoder can be defined in ``theanets`` by using the
    :class:`Autoencoder <theanets.feedforward.Autoencoder>` class::

      exp = theanets.Experiment(theanets.Autoencoder)

    The ``layers`` parameter is required to define such a model; it can be
    provided on the command-line by using ``--layers A B C ... A``, or in your
    code::

      exp = theanets.Experiment(
          theanets.Autoencoder,
          layers=(A, B, C, ..., A))

    Autoencoders retain all attributes of the parent :class:`Network` class,
    but additionally can have "tied weights", if the layer configuration is
    palindromic.

    Attributes
    ----------
    tied_weights : bool, optional
        Construct decoding weights using the transpose of the encoding weights
        on corresponding layers. Defaults to False, which means decoding weights
        will be constructed using a separate weight matrix.
    '''

    def setup_decoder(self):
        '''Set up weights for the decoder layers of an autoencoder.

        This implementation allows for decoding weights to be tied to encoding
        weights. If `tied_weights` is False, the decoder is set up using
        :func:`Network.setup_decoder`; if True, then the decoder is set up to be
        a mirror of the encoding layers, using transposed weights.

        Parameters
        ----------
        input_noise : float, optional
            Standard deviation of desired noise to inject into input.
        hidden_noise : float, optional
            Standard deviation of desired noise to inject into hidden unit
            activation output.
        input_dropouts : float in [0, 1], optional
            Proportion of input units to randomly set to 0.
        hidden_dropouts : float in [0, 1], optional
            Proportion of hidden unit activations to randomly set to 0.
        tied_weights : bool, optional
            If True, use decoding weights that are "tied" to the encoding
            weights. This only makes sense for a limited set of "autoencoder"
            layer configurations. Defaults to False.
        decode_from : int, optional
            For networks without tied weights, compute the activation of the
            output vector using the activations of the last N hidden layers in
            the network. Defaults to 1, which results in a traditional setup
            that decodes only from the penultimate layer in the network.

        Returns
        -------
        count : int
            A count of the number of tunable decoder parameters.
        '''
        if not self.tied_weights:
            return super(Autoencoder, self).setup_decoder()
        kw = {}
        kw.update(self.kwargs)
        kw.update(noise=self.kwargs.get('hidden_noise', 0),
                  dropout=self.kwargs.get('hidden_dropouts', 0))
        for i in range(len(self.layers) - 1, 1, -1):
            self.layers.append(layers.build('tied', self.layers[i], **kw))
        kw = {}
        kw.update(self.kwargs)
        kw.update(activation=self.output_activation)
        self.layers.append(layers.build('tied', self.layers[1], **kw))

    @property
    def encoding_layers(self):
        '''Compute the layers that will be part of the network encoder.

        This implementation ensures that --layers is compatible with
        --tied-weights; if so, and if the weights are tied, then the encoding
        layers are the first half of the layers in the network. If not, or if
        the weights are not to be tied, then all but the final layer is
        considered an encoding layer.

        Returns
        -------
        layers : list of int
            A list of integers specifying sizes of the encoder network layers.
        '''
        if not self.tied_weights:
            return super(Autoencoder, self).encoding_layers
        error = 'with --tied-weights, --layers must be an odd-length palindrome'
        sizes = []
        for layer in self.kwargs['layers']:
            if isinstance(layer, layers.Layer):
                sizes.append(layer.nout)
            if isinstance(layer, int):
                sizes.append(layer)
            if isinstance(layer, dict):
                sizes.append(layer.get('size', layer.get('nout', -1)))
        assert len(sizes) % 2 == 1, error
        k = len(sizes) // 2
        encode = np.asarray(sizes[:k])
        decode = np.asarray(sizes[k+1:])
        assert (encode == decode[::-1]).all(), error
        return self.kwargs['layers'][:k+1]

    @property
    def tied_weights(self):
        '''A boolean indicating whether this network uses tied weights.'''
        return self.kwargs.get('tied_weights', False)

    @property
    def error(self):
        '''Returns a theano expression for computing the mean squared error.'''
        err = self.outputs[-1] - self.x
        return TT.mean((err * err).sum(axis=1))

    def encode(self, x, layer=None, sample=False):
        '''Encode a dataset using the hidden layer activations of our network.

        Parameters
        ----------
        x : ndarray
            A dataset to encode. Rows of this dataset capture individual data
            points, while columns represent the variables in each data point.

        layer : int, optional
            The index of the hidden layer activation to use. By default, we use
            the "middle" hidden layer---for example, for a 4,2,4 or 4,3,2,3,4
            autoencoder, we use the "2" layer (index 1 or 2, respectively).

        sample : bool, optional
            If True, then draw a sample using the hidden activations as
            independent Bernoulli probabilities for the encoded data. This
            assumes the hidden layer has a logistic sigmoid activation function.

        Returns
        -------
        ndarray :
            The given dataset, encoded by the appropriate hidden layer
            activation.
        '''
        enc = self.feed_forward(x)[(layer or len(self.layers) // 2)]
        if sample:
            return np.random.binomial(n=1, p=enc).astype(np.uint8)
        return enc

    def decode(self, z, layer=None):
        '''Decode an encoded dataset by computing the output layer activation.

        Parameters
        ----------
        z : ndarray
            A matrix containing encoded data from this autoencoder.

        layer : int, optional
            The index of the hidden layer that was used to encode `z`.

        Returns
        -------
        ndarray :
            The decoded dataset.
        '''
        if not hasattr(self, '_decoders'):
            self._decoders = {}
        layer = layer or len(self.layers) // 2
        if layer not in self._decoders:
            outputs, _, updates = self._connect()
            self._decoders[layer] = theano.function(
                [outputs[layer]], [outputs[-1]], updates=updates)
        return self._decoders[layer](z)[0]


class Regressor(Network):
    r'''A regression model attempts to produce a target output.

    Regression models are trained by optimizing a (possibly regularized) loss
    that centers around some measurement of error with respect to the target
    outputs. This regression model implementation uses the mean squared error.

    If we have a labeled dataset containing :math:`m` :math:`d`-dimensional
    input samples :math:`X \in \mathbb{R}^{m \times d}` and :math:`m`
    :math:`e`-dimensional paired target outputs :math:`Y \in \mathbb{R}^{m
    \times e}`, then the loss that the Regressor model optimizes with respect to
    the model parameters :math:`\theta` is:

    .. math::
       \mathcal{L}(X, Y, \theta) = \frac{1}{m} \sum_{i=1}^m \| F_\theta(x_i) - y_i \|_2^2 + R(X, \theta)

    where :math:`F_\theta` is the feedforward function that computes the network
    output, and :math:`R` is a regularization function.
    '''

    def setup_vars(self):
        '''Setup Theano variables for our network.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        super(Regressor, self).setup_vars()

        # this variable holds the target outputs for input x.
        self.targets = TT.matrix('targets')

        return [self.x, self.targets]

    @property
    def error(self):
        '''Returns a theano expression for computing the mean squared error.'''
        err = self.outputs[-1] - self.targets
        return TT.mean((err * err).sum(axis=1))


class Classifier(Network):
    r'''A classifier attempts to match a 1-hot target output.

    Classification models in ``theanets`` are trained by optimizing a (possibly
    regularized) loss that centers around the categorical cross-entropy. This
    error computes the difference between the distribution generated by the
    classification model and the empirical distribution of the labeled data.

    If we have a labeled dataset containing :math:`m` :math:`d`-dimensional
    input samples :math:`X \in \mathbb{R}^{m \times d}` and :math:`m` paired
    target outputs :math:`Y \in \mathbb{R}^m`, then the loss that the
    ``Classifier`` model optimizes with respect to the model parameters
    :math:`\theta` is:

    .. math::
       \mathcal{L}(X, Y, \theta) = \frac{1}{m} \sum_{i=1}^m -\log F_\theta(x_i)_{y_i} + R(X, \theta)

    where :math:`F_\theta` is the softmax output generated by the classification
    model and :math:`R` is a regularization function.
    '''

    def setup_vars(self):
        '''Setup Theano variables for our network.

        Returns
        -------
        vars : list of theano variables
            A list of the variables that this network requires as inputs.
        '''
        super(Classifier, self).setup_vars()

        # for a classifier, this specifies the correct labels for a given input.
        self.labels = TT.ivector('labels')

        return [self.x, self.labels]

    @property
    def output_activation(self):
        return 'softmax'

    @property
    def error(self):
        '''Returns a theano computation of cross entropy.'''
        out = self.outputs[-1]
        prob = out[TT.arange(self.labels.shape[0]), self.labels]
        return -TT.mean(TT.log(prob))

    @property
    def accuracy(self):
        '''Returns a theano computation of percent correct classifications.'''
        out = self.outputs[-1]
        predict = TT.argmax(out, axis=1)
        return TT.cast(100, FLOAT) * TT.mean(TT.eq(predict, self.labels))

    @property
    def monitors(self):
        '''A sequence of name-value pairs for monitoring the network.

        Names in this sequence are strings, and values are theano variables
        describing how to compute the relevant quantity.

        These monitor expressions are used by network trainers to compute
        quantities of interest during training. The default set of monitors
        consists of everything from :func:`Network.monitors`, plus:

        - acc: the classification `accuracy` of the network
        '''
        for name, value in super(Classifier, self).monitors:
            yield name, value
        yield 'acc', self.accuracy

    def classify(self, x):
        '''Compute a greedy classification for the given set of data.

        Parameters
        ----------
        x : ndarray (num-examples, num-variables)
            An array containing examples to classify. Examples are given as the
            rows in this array.

        Returns
        -------
        k : ndarray (num-examples, )
            A vector of class index values, one per row of input data.
        '''
        return self.predict(x).argmax(axis=-1)
