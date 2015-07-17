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
import downhill
import fnmatch
import gzip
import hashlib
import numpy as np
import pickle
import theano
import theano.tensor as TT
import time
import warnings

from . import layers
from . import losses
from . import trainer

logging = climate.get_logger(__name__)


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
        If True, the network will require an additional input during training
        that provides weights for the target outputs of the network; the weights
        will be the last input argument to the network, and they must be the
        same shape as the target output.

        This can be particularly useful for recurrent networks, where the length
        of each input sequence in a minibatch is not necessarily the same number
        of time steps, or for classifier networks where the prior proabibility
        of one class is significantly different than another. The default is not
        to use weighted outputs.

    loss : str or :class:`Loss <losses.Loss>`
        The name of a loss function to optimize when training this network
        model.

    sparse_input : bool
        If True, create an input variable that can hold a sparse matrix.
        Defaults to False, which assumes all arrays are dense.

    Attributes
    ----------
    loss : :class:`Loss <losses.Loss>`
        A loss to be computed when optimizing this network model.
    layers : list of :class:`Layer <layers.Layer>`
        A list of the layers in this network model.
    '''

    def __init__(self, layers, loss='mse', weighted=False, sparse_input=False, **kwargs):
        self._graphs = {}     # cache of symbolic computation graphs
        self._functions = {}  # cache of callable feedforward functions
        self.loss = losses.build(
            loss, weighted=weighted, sparse_input=sparse_input, **kwargs)
        self.layers = []
        for i, layer in enumerate(layers):
            self.add_layer(layer, is_output=i == len(layers) - 1)
        logging.info('network has %d total parameters', self.num_params)

    def add_layer(self, layer, is_output=False):
        '''Add a layer to our network graph.

        Parameters
        ----------
        layer : int, tuple, dict, or :class:`Layer <layers.Layer>`
            A value specifying the layer to add. For more information, please
            see :ref:`creating-specifying-layers`.
        is_output : bool, optional
            True iff this is the output layer for the graph. This influences the
            default activation function used for the layer: output layers in
            most models have a linear activation, while output layers in
            classifier networks default to a softmax activation.
        '''
        # if the given layer is a Layer instance, just add it and move on.
        if isinstance(layer, layers.Layer):
            self.layers.append(layer)
            return

        # for the first layer, create an 'input' layer.
        if len(self.layers) == 0:
            assert isinstance(layer, int), 'first layer must be an int'
            self.layers.append(layers.build('input', size=layer, name='in'))
            return

        # here we set up some defaults for constructing a new layer.
        act = getattr(self, 'DEFAULT_OUTPUT_ACTIVATION', 'linear')
        form = 'feedforward'
        kwargs = dict(
            name='out' if is_output else 'hid{}'.format(len(self.layers)),
            activation=act if is_output else 'relu',
            inputs={self.layers[-1].output_name(): self.layers[-1].size},
            size=layer,
        )

        # if layer is a tuple, assume that it contains one or more of the following:
        # - a layers.Layer subclass to construct (type)
        # - the name of a layers.Layer class (str)
        # - the name of an activation function (str)
        # - the number of units in the layer (int)
        if isinstance(layer, (tuple, list)):
            for el in layer:
                try:
                    if issubclass(el, layers.Layer):
                        form = el.__name__
                except TypeError:
                    pass
                if isinstance(el, str):
                    if layers.Layer.is_registered(el):
                        form = el
                    else:
                        kwargs['activation'] = el
                if isinstance(el, int):
                    kwargs['size'] = el

        # if layer is a dictionary, try to extract a form for the layer, and
        # override our default keyword arguments with the rest.
        if isinstance(layer, dict):
            layer = dict(layer)
            if 'form' in layer:
                form = layer.pop('form').lower()
            kwargs.update(layer)

        if isinstance(form, str) and form.lower() == 'bidirectional':
            if not (isinstance(layer, dict) and 'name' in layer):
                kwargs['name'] = 'bd{}{}'.format(
                    kwargs.get('worker', 'rnn'), len(self.layers))

        if isinstance(form, str) and form.lower() == 'tied':
            partner = kwargs.get('partner')
            if isinstance(partner, str):
                # if the partner is named, just get that layer.
                partner = [l for l in self.layers if l.name == partner][0]
            else:
                # otherwise, we look backwards through our list of layers.
                # any "tied" layer that we find increases a counter by one,
                # and any "untied" layer decreases the counter by one. our
                # partner is the first layer we find with count zero.
                #
                # this is intended to handle the hopefully common case of a
                # (possibly deep) tied-weights autoencoder.
                tied = 1
                partner = None
                for l in self.layers[::-1]:
                    tied += 1 if isinstance(l, layers.Tied) else -1
                    if tied == 0:
                        partner = l
                        break
                assert partner is not None, \
                    'could not find tied layer partner for {} in {}'.format(
                        layer, self.layers)
            kwargs['partner'] = partner

        self.layers.append(layers.build(form, **kwargs))

    def itertrain(self, train, valid=None, algo='rmsprop', subalgo='rmsprop',
                  save_every=0, save_progress=None, **kwargs):
        '''Train our network, one batch at a time.

        This method yields a series of ``(train, valid)`` monitor pairs. The
        ``train`` value is a dictionary mapping names to monitor values
        evaluated on the training dataset. The ``valid`` value is also a
        dictionary mapping names to values, but these values are evaluated on
        the validation dataset.

        Because validation might not occur every training iteration, the
        validation monitors might be repeated for multiple training iterations.
        It is probably most helpful to think of the validation monitors as being
        the "most recent" values that have been computed.

        After training completes, the network attribute of this class will
        contain the trained network parameters.

        Parameters
        ----------
        train : :class:`Dataset <downhill.dataset.Dataset>` or list
            A dataset to use when training the network. If this is a
            ``downhill.Dataset`` instance, it will be used directly as the
            training datset. If it is a list of numpy arrays or a list of
            callables, it will be converted to a ``downhill.Dataset`` and then
            used as the training set.
        valid : :class:`Dataset <downhill.dataset.Dataset>` or list, optional
            If this is provided, it will be used as a validation dataset. If not
            provided, the training set will be used for validation. (This is not
            recommended!)
        algo : str, optional
            An optimization algorithm to use for training our network. If not
            provided, :class:`RMSProp <downhill.adaptive.RMSProp>` will be used.
        subalgo : str, optional
            An optimization algorithm to use for a trainer that requires a
            "sub-algorithm," sugh as an unsupervised pretrainer. Defaults to
            :class:`RMSProp <downhill.adaptive.RMSProp>`.
        save_every : int or float, optional
            If this is nonzero and ``save_progress`` is not None, then the model
            being trained will be saved periodically. If this is a float, it is
            treated as a number of minutes to wait between savings. If it is an
            int, it is treated as the number of training epochs to wait between
            savings. Defaults to 0.
        save_progress : str, optional
            If this is not None, and ``save_progress`` is nonzero, then save the
            model periodically during training. This parameter gives the full
            path of a file to save the model. If this name contains a "{}"
            format specifier, it will be filled with the integer Unix timestamp
            at the time the model is saved. Defaults to None.

        Yields
        ------
        training : dict
            A dictionary of monitor values computed using the training dataset,
            at the conclusion of training. This dictionary will at least contain
            a 'loss' key that indicates the value of the loss function. Other
            keys may be available depending on the trainer being used.
        validation : dict
            A dictionary of monitor values computed using the validation
            dataset, at the conclusion of training.
        '''
        def create_dataset(data, **kwargs):
            name = kwargs.get('name', 'dataset')
            s = '{}_batches'.format(name)
            return downhill.Dataset(
                data, name=name, batch_size=kwargs.get('batch_size', 32),
                iteration_size=kwargs.get('iteration_size', kwargs.get(s)),
                axis=kwargs.get('axis', 0))

        # set up datasets ...
        if valid is None:
            valid = train
        if not isinstance(valid, downhill.Dataset):
            valid = create_dataset(valid, name='valid', **kwargs)
        if not isinstance(train, downhill.Dataset):
            train = create_dataset(train, name='train', **kwargs)

        if 'algorithm' in kwargs:
            warnings.warn(
                'please use the "algo" keyword arg instead of "algorithm"',
                DeprecationWarning)
            algo = kwargs.pop('algorithm')
            if isinstance(algo, (list, tuple)):
                algo = algo[0]

        # set up trainer ...
        if isinstance(algo, str):
            algo = algo.lower()
            if algo == 'sample':
                algo = trainer.SampleTrainer(self)
            elif algo.startswith('layer') or algo.startswith('sup'):
                algo = trainer.SupervisedPretrainer(subalgo, self)
            elif algo.startswith('pre') or algo.startswith('unsup'):
                algo = trainer.UnsupervisedPretrainer(subalgo, self)
            else:
                algo = trainer.DownhillTrainer(algo, self)

        # set up check to save model ...
        def needs_saving(elapsed, iteration):
            if not save_progress:
                return False
            if isinstance(save_every, float):
                return elapsed > 60 * save_every
            if isinstance(save_every, int):
                return iteration % save_every == 0
            return False

        # train it!
        start = time.time()
        for i, monitors in enumerate(algo.itertrain(train, valid, **kwargs)):
            yield monitors
            now = time.time()
            if i and needs_saving(now - start, i):
                self.save(save_progress.format(int(now)))
                start = now

    def train(self, *args, **kwargs):
        '''Train the network until the trainer converges.

        All arguments are passed to :func:`itertrain`.

        Returns
        -------
        training : dict
            A dictionary of monitor values computed using the training dataset,
            at the conclusion of training. This dictionary will at least contain
            a 'loss' key that indicates the value of the loss function. Other
            keys may be available depending on the trainer being used.
        validation : dict
            A dictionary of monitor values computed using the validation
            dataset, at the conclusion of training.
        '''
        monitors = None
        for monitors in self.itertrain(*args, **kwargs):
            pass
        return monitors

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
        # See discussions
        # https://groups.google.com/forum/#!topic/theanets/nL6Nis29B7Q
        add(sorted(kwargs.items(), key=lambda x: x[0]))
        for l in self.layers:
            add('{}{}{}'.format(l.__class__.__name__, l.name, l.size))
        return h.hexdigest()

    def build_graph(self, **kwargs):
        '''Connect the layers in this network to form a computation graph.

        Parameters
        ----------
        noise : dict mapping str to float, optional
            A dictionary that maps layer output names to standard deviation
            values. For an output "layer:output" in the graph, white noise with
            the given standard deviation will be added to the output. Defaults
            to 0 for all layer outputs.
        dropout : dict mapping str to float in [0, 1], optional
            A dictionary that maps layer output names to dropout values. For an
            output "layer:output" in the graph, the given fraction of units in
            the output will be randomly set to 0. Default to 0 for all layer
            outputs.

        Returns
        -------
        outputs : list of theano variables
            A list of expressions giving the output of each layer in the graph.
        updates : list of update tuples
            A list of updates that should be performed by a theano function that
            computes something using this graph.
        '''
        key = self._hash(**kwargs)
        if key not in self._graphs:
            noise = kwargs.get('noise')
            if noise is None:
                noise = {}
                for i, l in enumerate(self.layers):
                    which = 'hidden_noise'
                    if i == 0:
                        which = 'input_noise'
                    if i == len(self.layers) - 1:
                        which = 'output_noise'
                    noise[l.output_name()] = kwargs.get(which, 0)
            dropout = kwargs.get('dropout')
            if dropout is None:
                dropout = {}
                for i, l in enumerate(self.layers):
                    which = 'hidden_dropouts'
                    if i == 0:
                        which = 'input_dropouts'
                    if i == len(self.layers) - 1:
                        which = 'output_dropouts'
                    dropout[l.output_name()] = kwargs.get(which, 0)
            outputs, updates = dict(x=self.loss.input), []
            for i, layer in enumerate(self.layers):
                out, upd = layer.connect(outputs, noise, dropout)
                outputs.update(out)
                updates.extend(upd)
                outputs['out'] = outputs[layer.output_name()]
            self._graphs[key] = outputs, updates
        return self._graphs[key]

    @property
    def params(self):
        '''A list of the learnable theano parameters for this network.'''
        return [p for l in self.layers for p in l.params]

    @property
    def num_params(self):
        '''Number of parameters in the entire network model.'''
        return sum(l.num_params for l in self.layers)

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
            outputs, updates = self.build_graph(**kwargs)
            labels, exprs = list(outputs.keys()), list(outputs.values())
            self._functions[key] = (labels, theano.function(
                [self.loss.input], exprs, updates=updates))
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
        y : ndarray (num-examples, num-variables)
            Returns the values of the network output units when given input `x`.
            Rows in this array correspond to examples, and columns to output
            variables.
        '''
        return self.feed_forward(x)[self.layers[-1].output_name()]

    def score(self, x, y, w=None):
        '''Compute R^2 coefficient of determination for a given labeled input.

        Parameters
        ----------
        x : ndarray (num-examples, num-inputs)
            An array containing data to be fed into the network. Multiple
            examples are arranged as rows in this array, with columns containing
            the variables for each example.
        y : ndarray (num-examples, num-outputs)
            An array containing expected target data for the network. Multiple
            examples are arranged as rows in this array, with columns containing
            the variables for each example.

        Returns
        -------
        r2 : float
            The R^2 correlation between the prediction of this netork and its
            target output.
        '''
        u = y - self.predict(x)
        v = y - y.mean()
        if w is None:
            w = np.ones_like(u)
        return 1 - (w * u * u).sum() / (w * v * v).sum()

    def save(self, filename):
        '''Save the state of this network to a pickle file on disk.

        Parameters
        ----------
        filename : str
            Save the state of this network to a pickle file at the named path.
            If this name ends in ".gz" then the output will automatically be
            gzipped; otherwise the output will be a "raw" pickle.
        '''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'wb')
        pickle.dump(self, handle, -1)
        handle.close()
        logging.info('%s: saved model', filename)

    @classmethod
    def load(cls, filename):
        '''Load a saved network from disk.

        Parameters
        ----------
        filename : str
            Load the state of a network from a pickle file at the named path. If
            this name ends in ".gz" then the input will automatically be
            gunzipped; otherwise the input will be treated as a "raw" pickle.
        '''
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'rb')
        model = pickle.load(handle)
        handle.close()
        logging.info('%s: loaded model', filename)
        return model

    def regularized_loss(self, **kwargs):
        '''Return a variable representing the regularized loss for this network.

        The regularized loss includes both the loss computation (the "error")
        for the network as well as any regularizers that are in place.

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
        noise : float, optional
            Standard deviation of desired noise to inject into input.
        dropout : float in [0, 1], optional
            Proportion of input units to randomly set to 0.

        Returns
        -------
        loss : theano expression
            A theano expression representing the loss of this network.
        '''
        outputs, _ = self.build_graph(**kwargs)
        hiddens = [outputs[l.output_name()] for l in self.layers[1:-1]]
        regularizers = dict(
            weight_l1=(abs(w).mean() for l in self.layers
                       for w in l.params if w.ndim > 1),
            weight_l2=((w * w).mean() for l in self.layers
                       for w in l.params if w.ndim > 1),
            hidden_l1=(abs(h).mean() for h in hiddens),
            hidden_l2=((h * h).mean() for h in hiddens),
            contractive=(TT.sqr(TT.grad(h.mean(), self.loss.input)).mean()
                         for h in hiddens),
        )
        out = outputs[self.layers[-1].output_name()]
        return self.loss(out) + sum(
            kwargs[weight] * sum(expr)
            for weight, expr in regularizers.items()
            if kwargs.get(weight, 0) > 0)

    def monitors(self, **kwargs):
        '''Return expressions that should be computed to monitor training.

        Returns
        -------
        monitors : list of (name, expression) pairs
            A list of named monitor expressions to compute for this network.
        '''
        outputs, _ = self.build_graph(**kwargs)
        out = outputs[self.layers[-1].output_name()]
        monitors = [('err', self.loss(out))]

        def parse_pattern(pattern):
            '''Yield graph expressions that match the given pattern.'''
            for name, expr in outputs.items():
                if fnmatch.fnmatch(name, pattern):
                    yield name, expr
            for l in self.layers:
                for p in l.params:
                    if fnmatch.fnmatch(p.name, pattern):
                        yield p.name, p

        def parse_levels(levels):
            '''Yield named monitor callables.'''
            if isinstance(levels, dict):
                levels = levels.items()
            if isinstance(levels, (int, float)):
                levels = [levels]
            for level in levels:
                if isinstance(level, (tuple, list)):
                    label, call = level
                    yield ':{}'.format(label), call
                if isinstance(level, (int, float)):
                    def call(expr):
                        return (expr < level).mean()
                    yield '<{}'.format(level), call

        inputs = kwargs.get('monitors', {})
        if isinstance(inputs, dict):
            inputs = inputs.items()
        for pattern, levels in inputs:
            for name, expr in parse_pattern(pattern):
                for key, value in parse_levels(levels):
                    monitors.append(('{}{}'.format(name, key), value(expr)))

        return monitors

    def updates(self, **kwargs):
        '''Return expressions to run as updates during network training.

        Returns
        -------
        updates : list of (parameter, expression) pairs
            A list of named parameter update expressions for this network.
        '''
        _, updates = self.build_graph(**kwargs)
        return updates
