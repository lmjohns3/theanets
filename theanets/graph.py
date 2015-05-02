# -*- coding: utf-8 -*-

r'''This module contains a base class for modeling computation graphs.

Neural networks are really just a concise, computational way of describing a
mathematical model of a computation graph that operates on a particular set of
data.

At a high level, a neural network is a computation graph that describes a
parametric mapping

.. math::
   F_\theta: \mathcal{S} \to \mathcal{T}

between a source space :math:`\mathcal{S}` and a target space
:math:`\mathcal{T}`, using parameters :math:`\theta`. For example, suppose we
are processing vectors representing the MNIST handwritten digits. We could think
of :math:`\mathcal{S} = \mathbb{R}^{28 \times 28} = \mathbb{R}^{784}` (i.e., the
space of all 28×28 images), and for classifying the MNIST digits we could think
of :math:`\mathcal{T} = \mathbb{R}^{10}`.

This mapping is assumed to be fairly complex. If it were not -- if you could
capture the mapping using a simple expression like :math:`F_{\{a\}}(x) = ax^2`
-- then we would just use the expression directly and not need to deal with an
entire network. So if the mapping is complex, we will do a couple of things to
make our problem tractable. First, we will assume some structure for
:math:`F_\theta`. Second, we will fit our model to some set of data that we have
obtained, so that our parameters :math:`\theta` are tuned to the problem at
hand.

Graph structure
---------------

.. image:: _static/feedforward_layers.svg

The mapping :math:`F_\theta` is implemented in neural networks by assuming a
specific, layered form. Computation nodes -- also called units or (sometimes)
neurons -- are arranged in a :math:`k+1` partite graph, with layer :math:`k`
containing :math:`n_k` nodes. The number of input nodes in the graph is referred
to as :math:`n_0`.

Most layers are connected together using a set of weights. A **weight matrix**
:math:`W^k \in \mathbb{R}^{n_{k-1} \times n_k}` specifies the strength of the
connection between nodes in layer :math:`k` and those in layer :math:`k-1` --
all other pairs of nodes are typically not connected. Each layer of nodes also
typically has a **bias vector** that determines the offset of each node from the
origin. Together, the parameters :math:`\theta` of the model are these :math:`k`
weight matrices and :math:`k` bias vectors (there are no weights or biases for
the input nodes in the graph).
'''

import climate
import gzip
import hashlib
import pickle
import theano
import theano.tensor as TT

from . import layers

logging = climate.get_logger(__name__)

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
    kw = dict(layers=pkl['layers'], weighted=pkl['weighted'])
    kw.update(kwargs)
    net = pkl['klass'](**kw)
    net.load_params(filename)
    return net


class Network(object):
    '''The network class encapsulates a network computation graph.

    In addition to defining standard functionality for common types of
    feedforward nets, there are also many options for specifying topology and
    regularization, several of which must be provided to the constructor at
    initialization time.

    Parameters
    ----------
    layers : sequence of int, tuple, dict, or :class:`Layer <layers.Layer>`
        A sequence of values specifying the layer configuration for the network.
        For more information, please see :ref:`creating-specifying-layers`.
    weighted : bool, optional
        If True, the network will require an additional input that provides
        weights for the target outputs of the network; the weights will be the
        last input argument to the network, and they must be the same shape as
        the target output. This can be particularly useful for recurrent
        networks, where the length of each input sequence in a minibatch is not
        necessarily the same number of time steps, or for classifier networks
        where the prior proabibility of one class is significantly different
        than another. The default is not to use weighted outputs.

    Attributes
    ----------
    layers : list of :class:`Layer <layers.Layer>`
        A list of the layers in this network model.
    weighted : bool
        True iff this network expects additional target weight inputs during
        training.
    '''

    def __init__(self, layers, weighted=False):
        self._graphs = {}     # cache of symbolic computation graphs
        self._functions = {}  # cache of callable feedforward functions
        self.weighted = weighted
        logging.info('constructing network')
        self.inputs = list(self.setup_vars())
        self.layers = self.setup_layers(layers)
        logging.info('network has %d total parameters', self.num_params)

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
        # x represents our network's input.
        self.x = TT.matrix('x')

        # the weight array is provided to ensure that different target values
        # are taken into account with different weights during optimization.
        self.weights = TT.matrix('weights')

        if self.weighted:
            return [self.x, self.weights]
        return [self.x]

    def setup_layers(self, specs):
        '''Set up a computation graph for our network.

        Subclasses may override this method to construct alternative network
        topologies.

        Parameters
        ----------
        specs : list of layer specifications
            A sequence of values specifying the layer configuration for the
            network.  For more information, please see
            :ref:`creating-specifying-layers`.

        Returns
        -------
        layers : list of :class:`Layer <layers.Layer>`
            A list of the layers for this network model.
        '''
        model = []
        for i, spec in enumerate(specs):
            # if spec is a Layer instance, just add it and move on.
            if isinstance(spec, layers.Layer):
                model.append(spec)
                continue

            # for the first layer, create an 'input' layer.
            if i == 0:
                model.append(layers.build('input', outputs=spec, name='in'))
                continue

            # here we set up some defaults for constructing a new layer.
            form = 'feedforward'
            prev = model[-1]
            last = i == len(specs) - 1
            kwargs = dict(
                name='out' if last else 'hid{}'.format(len(model)),
                activation='linear' if last else 'logistic',
                inputs={'{}.out'.format(prev.name): prev.outputs['out']},
                outputs=spec,
            )

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
                        kwargs['outputs'] = el
                kwargs['name'] = '{}{}'.format(form, len(model))

            # if spec is a dictionary, try to extract a form for the layer, and
            # override our default keyword arguments with the rest.
            if isinstance(spec, dict):
                if 'form' in spec:
                    form = spec['form'].lower()
                    kwargs['name'] = '{}{}'.format(form, len(model))
                kwargs.update(spec)

            if isinstance(form, str) and form.lower() == 'bidirectional':
                kwargs['name'] = 'bd{}{}'.format(
                    kwargs.get('worker', 'rnn'), len(model))

            if isinstance(form, str) and form.lower() == 'tied':
                partner = kwargs.get('partner')
                if isinstance(partner, str):
                    # if the partner is named, just get that layer.
                    partner = [l for l in model if l.name == partner][0]
                else:
                    # to find an unnamed partner for this tied layer, we look
                    # backwards through our list of layers. any "tied" layer
                    # that we find increases a counter by one, and any "untied"
                    # layer decreases the counter by one. our partner is the
                    # first layer we find with count zero.
                    #
                    # this is intended to handle the hopefully common case of a
                    # (possibly deep) tied-weights autoencoder.
                    tied = 0
                    partner = None
                    for l in model[::-1]:
                        tied += 1 if isinstance(l, layers.Tied) else -1
                        if tied == 0:
                            partner = l
                            break
                    assert partner is not None, \
                        'could not find tied layer partner: {}'.format(specs)
                kwargs['partner'] = partner

            model.append(layers.build(form, **kwargs))
        return model

    def error(self, output):
        '''Build a theano expression for computing the network error.

        Parameters
        ----------
        output : theano expression
            A theano expression representing the output of the network.

        Returns
        -------
        error : theano expression
            A theano expression representing the network error.
        '''
        err = output - self.x
        if self.weighted:
            return (self.weights * err * err).sum() / self.weights.sum()
        return (err * err).mean()

    def _hash(self, **kwargs):
        '''Construct a string key for representing a computation graph.

        This key will be unique for a given network topology and set of keyword
        arguments.

        Returns
        -------
        key : str
            A hash representing the computation graph for the current network.
        '''
        def add(s):
            h.update(str(s).encode('utf-8'))
        h = hashlib.md5()
        add(kwargs)
        for l in self.layers:
            add('{}{}{}'.format(
                l.__class__.__name__, l.name, sorted(l.outputs.items())))
        return h.hexdigest()

    def build_graph(self, **kwargs):
        '''Connect the layers in this network to form a computation graph.

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
        key = self._hash(**kwargs)
        if key not in self._graphs:
            inputs = dict(x=self.x)
            outputs, monitors, updates = {}, [], []
            for i, layer in enumerate(self.layers):
                noise = dropout = 0
                if i == 0:
                    noise = kwargs.get('input_noise', 0)
                    dropout = kwargs.get('input_dropouts', 0)
                elif i == len(self.layers) - 1:
                    noise = kwargs.get('hidden_noise', 0)
                    dropout = kwargs.get('hidden_dropouts', 0)
                out, mon, upd = layer.connect(inputs, noise=noise, dropout=dropout)
                inputs.update(out)
                scoped = {'.'.join((layer.name, n)): e for n, e in out.items()}
                inputs.update(scoped)
                outputs.update(scoped)
                monitors.extend(mon)
                updates.extend(upd)
            self._graphs[key] = outputs, monitors, updates
        return self._graphs[key]

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

    @property
    def num_params(self):
        '''Number of parameters in the entire network model.'''
        return sum(l.num_params for l in self.layers)

    @property
    def output_name(self):
        return self.layers[-1].output_name

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

    def feed_forward(self, x, **kwargs):
        '''Compute a forward pass of all layers from the given input.

        All keyword arguments are passed directly to :func:`build_graph`.

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
        key = self._hash(**kwargs)
        if key not in self._functions:
            outputs, _, updates = self.build_graph(**kwargs)
            labels, exprs = list(outputs.keys()), list(outputs.values())
            self._functions[key] = (
                labels,
                theano.function([self.x], exprs, updates=updates),
            )
        labels, f = self._functions[key]
        return dict(zip(labels, f(x)))

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
        return self.feed_forward(x)[self.output_name]

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
        state = dict(klass=self.__class__,
                     layers=[l.to_spec() for l in self.layers],
                     weighted=self.weighted)
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
        input_noise : float, optional
            Standard deviation of desired noise to inject into input.
        hidden_noise : float, optional
            Standard deviation of desired noise to inject into hidden unit
            activation output.
        input_dropouts : float in [0, 1], optional
            Proportion of input units to randomly set to 0.
        hidden_dropouts : float in [0, 1], optional
            Proportion of hidden unit activations to randomly set to 0.

        Returns
        -------
        loss : theano expression
            A theano expression representing the loss of this network.
        '''
        outputs, _, _ = self.build_graph(**kwargs)
        hiddens = [outputs[l.output_name] for l in self.layers[1:-1]]
        regularizers = dict(
            weight_l1=(abs(w).sum() for l in self.layers for w in l.params if w.ndim > 1),
            weight_l2=((w * w).sum() for l in self.layers for w in l.params if w.ndim > 1),
            hidden_l1=(abs(h).mean(axis=0).sum() for h in hiddens),
            hidden_l2=((h * h).mean(axis=0).sum() for h in hiddens),
            contractive=(TT.sqr(TT.grad(h.mean(axis=0).sum(), self.x)).sum()
                         for h in hiddens),
        )
        return self.error(outputs[self.output_name]) + sum(
            TT.cast(kwargs[weight], FLOAT) * sum(expr)
            for weight, expr in regularizers.items()
            if kwargs.get(weight, 0) > 0)

    def monitors(self, **kwargs):
        '''Return expressions that should be computed to monitor training.

        Returns
        -------
        monitors : list of (name, expression) pairs
            A list of named monitor expressions to compute for this network.
        '''
        outputs, monitors, _ = self.build_graph(**kwargs)
        return [('err', self.error(outputs[self.output_name]))] + monitors

    def updates(self, **kwargs):
        '''Return expressions to run as updates during network training.

        Returns
        -------
        updates : list of (parameter, expression) pairs
            A list of named parameter update expressions for this network.
        '''
        _, _, updates = self.build_graph(**kwargs)
        return updates
