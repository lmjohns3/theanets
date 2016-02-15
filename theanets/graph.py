# -*- coding: utf-8 -*-

r'''This module contains a base class for modeling computation graphs.'''

import climate
import downhill
import gzip
import hashlib
import numpy as np
import pickle
import theano
import time
import warnings

from . import layers
from . import losses
from . import regularizers
from . import trainer
from . import util

logging = climate.get_logger(__name__)


class Network(object):
    '''The network class encapsulates a network computation graph.

    Notes
    -----

    Computation graphs are organized into :ref:`layers <layers>`. Each layer
    receives one or more arrays of input data, transforms them, and generates
    one or more arrays of output data.

    Outputs in a computation graph are named according to their layer and output
    type, so the 'pre' output of a layer named 'hid1' would be named 'hid1:pre'.
    The 'out' output is the default output for a layer. By default the last
    layer in a network is named 'out'.

    The parameters in a network graph are optimized by minimizing a :ref:`loss
    function <losses>` with respect to some set of training data. Typically the
    value produced by 'out:out' is compared to some target value, creating an
    error value of some sort. This error value is then propagated back through
    the computation graph to update the parameters in the model.

    Parameters
    ----------
    layers : int, tuple, dict, or :class:`Layer <theanets.layers.base.Layer>`
        A sequence of values specifying the layer configuration for the network.
        For more information, please see :ref:`guide-creating-specifying-layers`.
    loss : str or :class:`Loss <theanets.losses.Loss>`
        The name of a loss function to optimize when training this network
        model.
    weighted : bool, optional
        If True, optimize this model using a "weighted" loss. Weighted losses
        typically require an additional array as input during optimization.
        For more information, see :ref:`losses-weighted`. Defaults to False.
    rng : int or RandomState, optional
        A seed or numpy ``RandomState`` instance for generating randomness in
        the model. Defaults to 13.

    Attributes
    ----------
    layers : list of :class:`Layer <theanets.layers.base.Layer>`
        A list of the layers in this network model.
    losses : list of :class:`Loss <theanets.losses.Loss>`
        A list of losses to be computed when optimizing this network model.
    '''

    DEFAULT_OUTPUT_ACTIVATION = 'linear'
    '''Default activation for the output layer.'''

    INPUT_NDIM = 2
    '''Number of dimensions for holding input data arrays.'''

    OUTPUT_NDIM = 2
    '''Number of dimensions for holding output data arrays.'''

    def __init__(self, layers=(), loss='mse', weighted=False, rng=13):
        self._graphs = {}     # cache of symbolic computation graphs
        self._functions = {}  # cache of callable feedforward functions
        self._rng = rng

        self.layers = []
        for i, layer in enumerate(layers):
            self.add_layer(layer=layer, is_output=i == len(layers) - 1)
        [l.bind(self) for l in self.layers]

        self.losses = []
        if loss and self.layers:
            self.set_loss(loss,
                          weighted=weighted,
                          target=self.OUTPUT_NDIM,
                          output_name=self.layers[-1].output_name)

    def add_layer(self, layer=None, is_output=False, **kwargs):
        '''Add a :ref:`layer <layers>` to our network graph.

        Parameters
        ----------
        layer : int, tuple, dict, or :class:`Layer <theanets.layers.base.Layer>`
            A value specifying the layer to add. For more information, please
            see :ref:`guide-creating-specifying-layers`.
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

        form = kwargs.pop('form', 'ff' if self.layers else 'input').lower()

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
                if isinstance(el, util.basestring):
                    if layers.Layer.is_registered(el):
                        form = el
                    else:
                        kwargs['activation'] = el
                if isinstance(el, int):
                    kwargs['size'] = el

        # if layer is a dictionary, try to extract a form for the layer, and
        # override our default keyword arguments with the rest.
        if isinstance(layer, dict):
            for key, value in layer.items():
                if key == 'form':
                    form = value.lower()
                else:
                    kwargs[key] = value

        name = 'hid{}'.format(len(self.layers))
        if is_output:
            name = 'out'
        if form == 'input':
            name = 'in'
        kwargs.setdefault('name', name)
        kwargs.setdefault('size', layer)

        if form == 'input':
            kwargs.setdefault('ndim', self.INPUT_NDIM)
        else:
            act = self.DEFAULT_OUTPUT_ACTIVATION if is_output else 'relu'
            kwargs.setdefault('inputs', self.layers[-1].output_name)
            kwargs.setdefault('rng', self._rng)
            kwargs.setdefault('activation', act)

        if form.lower() == 'tied' and 'partner' not in kwargs:
            # we look backward through our list of layers for a partner.
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
                    partner = l.name
                    break
            else:
                raise util.ConfigurationError(
                    'cannot find partner for "{}"'.format(kwargs))
            kwargs['partner'] = partner

        layer = layers.Layer.build(form, **kwargs)

        if isinstance(layer, layers.Input):
            names = set(i.name for i in self.inputs)
            assert layer.name not in names, \
                '"{}": duplicate input name!'.format(layer.name)

        self.layers.append(layer)

    def add_loss(self, loss=None, **kwargs):
        '''Add a :ref:`loss function <losses>` to the model.

        Parameters
        ----------
        loss : str, dict, or :class:`theanets.losses.Loss`
            A loss function to add. If this is a Loss instance, it will be added
            immediately. If this is a string, it names a loss function to build
            and add. If it is a dictionary, it should contain a ``'form'`` key
            whose string value names the loss function to add. Other arguments
            will be passed to :func:`theanets.losses.Loss.build`.
        '''
        if isinstance(loss, losses.Loss):
            self.losses.append(loss)
            return

        form = loss or 'mse'
        if 'form' in kwargs:
            form = kwargs.pop('form').lower()

        kw = dict(target=self.INPUT_NDIM, output_name=self.layers[-1].output_name)
        kw.update(kwargs)

        if isinstance(loss, dict):
            loss = dict(loss)
            if 'form' in loss:
                form = loss.pop('form').lower()
            kw.update(loss)

        self.losses.append(losses.Loss.build(form, **kw))

    def set_loss(self, *args, **kwargs):
        '''Clear the current loss functions from the network and add a new one.

        All parameters and keyword arguments are passed to :func:`add_loss`
        after clearing the current losses.
        '''
        self.losses = []
        self.add_loss(*args, **kwargs)

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
        if 'rng' not in kwargs:
            kwargs['rng'] = self._rng

        def create_dataset(data, **kwargs):
            name = kwargs.get('name', 'dataset')
            s = '{}_batches'.format(name)
            return downhill.Dataset(
                data,
                name=name,
                batch_size=kwargs.get('batch_size', 32),
                iteration_size=kwargs.get('iteration_size', kwargs.get(s)),
                axis=kwargs.get('axis', 0),
                rng=kwargs['rng'])

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
        if isinstance(algo, util.basestring):
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

    def _hash(self, regularizers=()):
        '''Construct a string key for representing a computation graph.

        This key will be unique for a given (a) network topology, (b) set of
        losses, and (c) set of regularizers.

        Returns
        -------
        key : str
            A hash representing the computation graph for the current network.
        '''
        def add(s):
            h.update(str(s).encode('utf-8'))
        h = hashlib.md5()
        for l in self.layers:
            add('{}{}{}'.format(l.__class__.__name__, l.name, l.size))
        for l in self.losses:
            add('{}{}'.format(l.__class__.__name__, l.weight))
        for r in regularizers:
            add('{}{}{}'.format(r.__class__.__name__, r.weight, r.pattern))
        return h.hexdigest()

    def build_graph(self, regularizers=()):
        '''Connect the layers in this network to form a computation graph.

        Parameters
        ----------
        regularizers : list of :class:`theanets.regularizers.Regularizer`
            A list of the regularizers to apply while building the computation
            graph.

        Returns
        -------
        outputs : list of Theano variables
            A list of expressions giving the output of each layer in the graph.
        updates : list of update tuples
            A list of updates that should be performed by a Theano function that
            computes something using this graph.
        '''
        key = self._hash(regularizers)
        if key not in self._graphs:
            logging.info('building computation graph')
            for loss in self.losses:
                loss.log()
            for reg in regularizers:
                reg.log()
            outputs = {}
            updates = []
            for layer in self.layers:
                out, upd = layer.connect(outputs)
                for reg in regularizers:
                    reg.modify_graph(out)
                outputs.update(out)
                updates.extend(upd)
            self._graphs[key] = outputs, updates
        return self._graphs[key]

    @property
    def inputs(self):
        '''A list of Theano variables for feedforward computations.'''
        return [l.input for l in self.layers if isinstance(l, layers.Input)]

    @property
    def variables(self):
        '''A list of Theano variables for loss computations.'''
        result = self.inputs
        seen = set(i.name for i in result)
        for loss in self.losses:
            for v in loss.variables:
                if v.name not in seen:
                    result.append(v)
                    seen.add(v.name)
        return result

    @property
    def params(self):
        '''A list of the learnable Theano parameters for this network.'''
        return [p for l in self.layers for p in l.params]

    def find(self, which, param):
        '''Get a parameter from a layer in the network.

        Parameters
        ----------
        which : int or str
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
        param : Theano shared variable
            A shared parameter variable from the indicated layer.
        '''
        for i, layer in enumerate(self.layers):
            if which == i or which == layer.name:
                return layer.find(param)
        raise KeyError(which)

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
        regs = regularizers.from_kwargs(self, **kwargs)
        key = self._hash(regs)
        if key not in self._functions:
            outputs, updates = self.build_graph(regs)
            labels, exprs = list(outputs.keys()), list(outputs.values())
            logging.info('compiling feed_forward function')
            self._functions[key] = (labels, theano.function(
                self.inputs, exprs, updates=updates))
        labels, f = self._functions[key]
        return dict(zip(labels, f(x)))

    def predict(self, x, **kwargs):
        '''Compute a forward pass of the inputs, returning the network output.

        All keyword arguments end up being passed to :func:`build_graph`.

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
        return self.feed_forward(x, **kwargs)[self.layers[-1].output_name]

    def score(self, x, y, w=None, **kwargs):
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
        u = y - self.predict(x, **kwargs)
        v = y - y.mean()
        if w is None:
            w = np.ones_like(u)
        return 1 - (w * u * u).sum() / (w * v * v).sum()

    def __getstate__(self):
        return (self.layers, self.losses)

    def __setstate__(self, state):
        self.layers, self.losses = state
        self._graphs = {}
        self._functions = {}

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

    def loss(self, **kwargs):
        '''Return a variable representing the regularized loss for this network.

        The regularized loss includes both the :ref:`loss computation <losses>`
        for the network as well as any :ref:`regularizers <regularizers>` that
        are in place.

        Keyword arguments are passed directly to
        :func:`theanets.regularizers.from_kwargs`.

        Returns
        -------
        loss : Theano expression
            A Theano expression representing the loss of this network.
        '''
        regs = regularizers.from_kwargs(self, **kwargs)
        outputs, _ = self.build_graph(regs)
        return sum(l.weight * l(outputs) for l in self.losses) + \
            sum(r.weight * r.loss(self.layers, outputs) for r in regs)

    def monitors(self, **kwargs):
        '''Return expressions that should be computed to monitor training.

        Returns
        -------
        monitors : list of (name, expression) pairs
            A list of named monitor expressions to compute for this network.
        '''
        regs = regularizers.from_kwargs(self, **kwargs)
        outputs, _ = self.build_graph(regs)
        monitors = [('err', self.losses[0](outputs))]

        def matching(pattern):
            '''Yield all matching outputs or parameters from the graph.'''
            for name, expr in util.outputs_matching(outputs, pattern):
                yield name, expr
            for name, expr in util.params_matching(self.layers, pattern):
                yield name, expr

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
            for name, expr in matching(pattern):
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
        regs = regularizers.from_kwargs(self, **kwargs)
        _, updates = self.build_graph(regs)
        return updates
