#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import theanets

import utils

COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e',
          '#e377c2', '#8c564b', '#bcbd22', '#7f7f7f', '#17becf']

URL = 'http://www.gutenberg.org/cache/epub/2701/pg2701.txt'

with open(utils.find('moby.txt', URL)) as handle:
    text = theanets.recurrent.Text(handle.read().lower().replace('\n', ' '))

seed = text.encode(text.text[200000:200010])
for i, layer in enumerate((
        dict(form='rnn', activation='sigmoid', diagonal=0.99),
        dict(form='gru', activation='sigmoid'),
        dict(form='scrn', activation='sigmoid'),
        dict(form='bcrnn', activation='sigmoid', num_modules=5),
        dict(form='lstm'),
        dict(form='mrnn', activation='sigmoid', factors=len(text.alpha)),
        dict(form='clockwork', activation='sigmoid', periods=(1, 2, 4, 8, 16)))):
    losses = []
    layer.update(size=100)
    net = theanets.recurrent.Classifier([
        1 + len(text.alpha), layer, 1000, 1 + len(text.alpha)])
    for tm, _ in net.itertrain(text.classifier_batches(30, 16),
                               min_improvement=0.99,
                               validate_every=50,
                               patience=0,
                               algo='adam',
                               max_gradient_norm=1,
                               learning_rate=0.01):
        if np.isnan(tm['loss']):
            break
        print('{}|{} ({:.1f}%)'.format(
            text.decode(seed),
            text.decode(net.predict_sequence(seed, 30)),
            100 * tm['acc']))
        losses.append(tm['loss'])

    plt.plot(losses, label=layer['form'], alpha=0.7, color=COLORS[i])

plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['bottom'].set_position(('outward', 6))
plt.gca().spines['left'].set_position(('outward', 6))

plt.gca().set_ylabel('Loss')
plt.gca().set_xlabel('Training Epoch')
plt.gca().grid(True)

plt.legend()
plt.show()
