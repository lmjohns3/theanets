#!/usr/bin/env python

import climate
import cPickle as pickle
import gzip
import numpy as np

logging = climate.get_logger('theanets-untie')

@climate.annotate(
    source='load a saved network from FILE',
    target='save untied network weights to FILE',
)
def main(source, target):
    opener = gzip.open if source.endswith('.gz') else open
    p = pickle.load(opener(source))

    logging.info('read from %s:', source)
    for w, b in zip(p['weights'], p['biases']):
        logging.info('weights %s bias %s %s', w.shape, b.shape, b.dtype)

    p['weights'].extend(0 + w.T for w in p['weights'][::-1])
    p['biases'].extend(-b for b in p['biases'][-2::-1])
    p['biases'].append(np.zeros(
        (len(p['weights'][0]), ), p['biases'][0].dtype))

    logging.info('writing to %s:', target)
    for w, b in zip(p['weights'], p['biases']):
        logging.info('weights %s bias %s %s', w.shape, b.shape, b.dtype)

    opener = gzip.open if target.endswith('.gz') else open
    pickle.dump(p, opener(target, 'wb'), -1)


if __name__ == '__main__':
    climate.call(main)
