'''This package groups together a bunch of theano code for neural nets.'''

from .feedforward import Autoencoder, Regressor, Classifier
from .graph import Network
from .main import Experiment

from . import layers
from . import recurrent
from . import trainer

__version__ = '0.6.3'
