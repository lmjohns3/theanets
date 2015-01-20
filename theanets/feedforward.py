# Copyright (c) 2012-2014 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''This module contains a number of classes for modeling neural nets in Theano.
'''

import climate
import functools
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
    layers : sequence of int
        A sequence of integers specifying the number of units at each layer. As
        an example, layers=(10, 20, 3) has one "input" layer with 10 units, one
        "hidden" layer with 20 units, and one "output" layer with 3 units. That
        is, inputs should be of length 10, and outputs will be of length 3.

    hidden_activation : str, optional
        The name of an activation function to use on hidden network units.
        Defaults to 'sigmoid'.

    output_activation : str, optional
        The name of an activation function to use on output units. Defaults to
        'linear'.

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
    layers : list of :class:`theanets.Layer`

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

        specs = list(self.encode_layers)
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

        logging.info('%d total network parameters',
                     sum(l.reset() for l in self.layers))

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
    def encode_layers(self):
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
        updates : list of update tuples
            A list of updates that should be performed by a theano function that
            computes something using this graph.
        '''
        outputs = []
        updates = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                # input to first layer is data.
                inputs = (self.x, )
            elif i == len(self.layers) - 1:
                # inputs to last layer is output of layers to decode.
                inputs = outputs[-self.kwargs.get('decode_from', 1):]
            else:
                # inputs to other layers are outputs of previous layer.
                inputs = outputs[-1:]
            out, upd = layer.output(inputs)
            outputs.append(out)
            updates.extend(upd)
        return outputs, updates

    @property
    def outputs(self):
        return self._connect()[0]

    @property
    def updates(self):
        return self._connect()[1]

    @property
    def monitors(self):
        '''A sequence of name-value pairs for monitoring the network.

        Names in this sequence are strings, and values are theano variables
        describing how to compute the relevant quantity.

        These monitor expressions are used by network trainers to compute
        quantities of interest during training. The default set of monitors
        consists of:

        - err: the unregularized error of the network
        - X<0.1: percent of hidden units in layer X such that :math:`|a_i| < 0.1`
        - X<0.9: percent of hidden units in layer X such that :math:`|a_i| < 0.9`
        '''
        yield 'err', self.error
        for i, (layer, output) in enumerate(zip(self.layers, self.outputs)):
            yield '{}<0.1'.format(layer.name), 100 * (abs(output) < 0.1).mean()
            yield '{}<0.9'.format(layer.name), 100 * (abs(output) < 0.9).mean()

    def params(self, **kwargs):
        '''Get a list of the learnable theano parameters for this network.

        This method is used internally by :class:`theanets.trainer.Trainer`
        implementations to compute the set of parameters that are tunable in a
        network.

        Returns
        -------
        params : list of theano variables
            A list of parameters that can be learned in this model.
        '''
        exclude_bias = kwargs.get('no_learn_bias', False)
        params = []
        for layer in self.layers:
            params.extend(layer.get_params(exclude_bias=exclude_bias))
        return params

    def get_layer(self, which):
        '''Return the current weights for a given layer.

        Parameters
        ----------
        which : int or str
            The layer of weights to return. If this is an integer, then 1 refers
            to the "first" hidden layer, 2 to the "second", and so on. If it is
            a string, the layer with the corresponding name, if any, will be
            used.

        Raises
        ------
        KeyError
            If there is no such layer.

        Returns
        -------
        layer : :class:`layers.Layer`
            The layer in the network with this name or index.
        '''
        try:
            if isinstance(which, int):
                return self.layers[which]
            return [l for l in self.layers if l.name == which][0]
        except:
            raise KeyError(which)

    def get_weights(self, which, index=0, borrow=False):
        '''Return the current weights for a given layer.

        Parameters
        ----------
        which : int or str
            The layer of weights to return. If this is an integer, then 1 refers
            to the "first" hidden layer, 2 to the "second", and so on. If it is
            a string, the layer with the corresponding name, if any, will be
            used.
        index : int, optional
            Index of the weights to get from this layer. Most layers just have
            one set of weights, so this defaults to 0. Recurrent layers might
            have many sets, however, and the index will depend on the
            implementation.
        borrow : bool, optional
            Whether to "borrow" the reference to the weights. If True, this
            returns a view onto the current weight array; if False (default), it
            returns a copy of the weight array.

        Raises
        ------
        KeyError
            If there is no such layer.
        IndexError
            If there is no such weight array in the given layer.

        Returns
        -------
        weights : ndarray
            The weight values, as a numpy array.

        '''
        return self.get_layer(which).weights[index].get_value(borrow=borrow)

    def get_bias(self, layer, index=0, borrow=False):
        '''Return the current bias vector for a given layer.

        Parameters
        ----------
        which : int or str
            The layer of weights to return. If this is an integer, then 1 refers
            to the "first" hidden layer, 2 to the "second", and so on. If it is
            a string, the layer with the corresponding name, if any, will be
            used.
        index : int, optional
            Index of the bias values to get from this layer. Most layers just
            have one set of bias values, so this defaults to 0. Recurrent layers
            might have many sets, however, and the index will depend on the
            implementation.
        borrow : bool, optional
            Whether to "borrow" the reference to the biases. If True, this
            returns a view onto the current bias vector; if False (default), it
            returns a copy of the biases.

        Raises
        ------
        KeyError
            If there is no such layer.
        IndexError
            If there is no such bias vector in the given layer.

        Returns
        -------
        bias : ndarray
            The bias values, as a numpy vector.
        '''
        return self.get_layer(which).biases[index].get_value(borrow=borrow)

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
            outputs, updates = self._connect()
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
            state['{}-values'.format(layer.name)] = layer.get_values()
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
            layer.set_values(saved['{}-values'.format(layer.name)])
        logging.info('%s: loaded model parameters', filename)

    def loss(self, weight_l1=0, weight_l2=0, hidden_l1=0, hidden_l2=0, contractive_l2=0, **unused):
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
        contractive_l2 : float, optional
            Regularize model using the Frobenius norm of the hidden Jacobian.

        Returns
        -------
        loss : theano variable
            A variable representing the loss of this network.
        '''
        outputs, _ = self._connect()
        hiddens = outputs[1:-1]
        loss = self.error
        if weight_l1 > 0:
            loss += weight_l1 * sum(abs(w).sum() for l in self.layers for w in l.weights)
        if weight_l2 > 0:
            loss += weight_l2 * sum((w * w).sum() for l in self.layers for w in l.weights)
        if hidden_l1 > 0:
            loss += hidden_l1 * sum(abs(h).mean(axis=0).sum() for h in hiddens)
        if hidden_l2 > 0:
            loss += hidden_l2 * sum((h * h).mean(axis=0).sum() for h in hiddens)
        if contractive_l2 > 0:
            loss += contractive_l2 * sum(
                TT.sqr(TT.grad(h.mean(axis=0).sum(), self.x)).sum() for h in hiddens)
        return loss


class Autoencoder(Network):
    '''An autoencoder attempts to reproduce its input.

    Autoencoders retain all attributes of the parent class (:class:`Network`),
    but additionally can have "tied weights".

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
    def encode_layers(self):
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
            return super(Autoencoder, self).encode_layers
        error = 'with --tied-weights, --layers must be an odd-length palindrome'
        sizes = self.kwargs['layers']
        assert len(sizes) % 2 == 1, error
        k = len(sizes) // 2
        encode = np.asarray(sizes[:k])
        decode = np.asarray(sizes[k+1:])
        assert (encode == decode[::-1]).all(), error
        return sizes[:k+1]

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
            outputs, updates = self._connect()
            self._decoders[layer] = theano.function(
                [outputs[layer]], [outputs[-1]], updates=updates)
        return self._decoders[layer](z)[0]


class Regressor(Network):
    '''A regressor attempts to produce a target output.'''

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
    '''A classifier attempts to match a 1-hot target output.'''

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
        return 'logsoftmax'

    @property
    def error(self):
        '''Returns a theano computation of cross entropy.'''
        out = self.outputs[-1]  # flatten all but last components of the output below, also flatten the labels
        idx = TT.arange(TT.prod(self.labels.shape))
        logprobs = TT.reshape(out, (TT.prod(out.shape[:-1]), out.shape[-1]))
        return -TT.mean(logprobs[idx, self.labels.flatten(1)])

    @property
    def accuracy(self):
        '''Returns a theano computation of percent correct classifications.'''
        out = self.outputs[-1]
        probs = TT.reshape(out, (TT.prod(out.shape[:-1]), out.shape[-1]))
        return 100 * TT.mean(TT.eq(TT.argmax(probs, axis=-1), self.labels.flatten(1)))

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
