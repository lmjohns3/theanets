#!/usr/bin/env python

import logging
import numpy as np

import lmj.nn

lmj.nn.enable_default_logging()

class Main(lmj.nn.Main):
    def get_network(self):
        return lmj.nn.recurrent.Autoencoder

    def get_datasets(self):
        t = np.linspace(0, 4 * np.pi, 256)
        train = np.array([np.sin(t + i) for i in range(64, 256)])
        dev = np.array([np.sin(t + i) for i in range(64)])
        return train, dev

m = Main(layers=(1, 3, 1), batch_size=1)
m.train()
