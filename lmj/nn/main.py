# Copyright (c) 2012 Leif Johnson <leif@leifjohnson.net>
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

'''This file contains an object encapsulating a main process.'''

import lmj.cli
import theano.tensor as TT

from .dataset import SequenceDataset as Dataset
from . import trainer

logging = lmj.cli.get_logger(__name__)


class Main(object):
    '''This class sets up the infrastructure to train a net.

    Two methods must be implemented by subclasses -- get_network must return the
    Network subclass to instantiate, and get_datasets must return a tuple of
    training and validation datasets. Subclasses have access to command line
    arguments through self.args.
    '''

    def __init__(self, args=None, **kwargs):
        self.args = args or lmj.cli.get_args().parse_args()
        for k, v in kwargs.iteritems():
            setattr(self.args, k, v)

        kwargs = {}
        kwargs.update(vars(self.args))
        logging.info('command-line options:')
        for k in sorted(kwargs):
            logging.info('--%s = %s', k, kwargs[k])

        activation = self.get_activation()
        if hasattr(activation, 'lmj_nn_name'):
            logging.info('activation: %s', activation.lmj_nn_name)

        self.net = self.get_network()(
            layers=self.args.layers,
            activation=activation,
            decode=self.args.decode,
            tied_weights=self.args.tied_weights,
            input_noise=self.args.input_noise,
            hidden_noise=self.args.hidden_noise,
            input_dropouts=self.args.input_dropouts,
            hidden_dropouts=self.args.hidden_dropouts,
            damping=self.args.damping,
            )

        kw = dict(size=self.args.batch_size)
        train_, valid_ = tuple(self.get_datasets())[:2]
        if not isinstance(train_, (tuple, list)):
            train_ = (train_, )
        if not isinstance(valid_, (tuple, list)):
            valid_ = (valid_, )

        kw.update(dict(batches=self.args.train_batches, label='train'))
        self.train_set = Dataset(*train_, **kw)

        kw.update(dict(batches=self.args.valid_batches, label='valid'))
        self.valid_set = Dataset(*valid_, **kw)

        kw.update(dict(batches=self.args.cg_batches, label='cg'))
        kwargs['cg_set'] = Dataset(*train_, **kw)

        self.trainer = self.get_trainer()(self.net, **kwargs)

    def train(self):
        self.trainer.train(self.train_set, self.valid_set)

    def get_activation(self, act=None):
        def compose(a, b):
            c = lambda z: b(a(z))
            c.lmj_nn_name = '%s(%s)' % (b.lmj_nn_name, a.lmj_nn_name)
            return c
        act = act or self.args.activation.lower()
        if '+' in act:
            return reduce(compose, (self.get_activation(a) for a in act.split('+')))
        options = {
            'tanh': TT.tanh,
            'linear': lambda z: z,
            'logistic': TT.nnet.sigmoid,

            # shorthands
            'relu': lambda z: TT.maximum(0, z),

            # modifiers
            'rect:max': lambda z: TT.minimum(1, z),
            'rect:min': lambda z: TT.maximum(0, z),

            # normalization
            'norm:dc': lambda z: z - z.mean(axis=1)[:, None],
            'norm:max': lambda z: z / TT.maximum(1e-10, abs(z).max(axis=1)[:, None]),
            'norm:std': lambda z: z / TT.maximum(1e-10, z.std(axis=1)[:, None]),
            }
        for k, v in options.iteritems():
            v.lmj_nn_name = k
        try:
            return options[act]
        except:
            raise KeyError('unknown --activation %s' % act)

    def get_trainer(self, opt=None):
        opt = opt or self.args.optimize.lower()
        if '+' in opt:
            return trainer.Cascaded(self.get_trainer(o) for o in opt.split('+'))
        try:
            return {
                'hf': trainer.HF,
                'sgd': trainer.SGD,
                'sample': trainer.Sample,
                'layerwise': trainer.Layerwise,
                'force': trainer.FORCE,
                }[opt]
        except:
            raise KeyError('unknown --optimize %s' % opt)

    def get_network(self):
        raise NotImplementedError

    def get_datasets(self):
        raise NotImplementedError
