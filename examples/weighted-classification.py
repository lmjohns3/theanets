import climate
import numpy as np
import sklearn.datasets
import sklearn.metrics
import theanets

climate.enable_default_logging()

samples, labels = sklearn.datasets.make_classification(
    n_samples=10000,
    n_features=100,
    n_informative=30,
    n_redundant=30,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=3,
    weights=[0.99, 0.01],
    flip_y=0.01,
)

weights = np.ones_like(labels)
weights[labels.nonzero()] *= 10


def split(a, b):
    return [samples[a:b].astype('float32'),
            labels[a:b].astype('int32'),
            weights[a:b].astype('float32')]

train = split(0, 9000)
valid = split(9000, 10000)

net = theanets.Classifier(
    layers=(100, 10, 2),
    weighted=True,
)

net.train(train, valid)

truth = valid[1]
print('# of true 1s:', truth.sum())

guess = net.predict(valid[0])
print('# of predicted 1s:', guess.sum())

cm = sklearn.metrics.confusion_matrix(truth, guess)
print('confusion matrix (true class = rows, predicted class = cols):')
print(cm)
