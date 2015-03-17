import climate
import numpy as np
import sklearn.datasets
import theanets

climate.enable_default_logging()

samples, labels = sklearn.datasets.make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    flip_y=0.01,
)

print(labels)

weights = np.ones_like(labels)
weights[labels.nonzero()] *= 1

exp = theanets.Experiment(
    theanets.Classifier,
    layers=(20, 10, 2),
    weighted=True,
)

exp.train([samples.astype('float32'),
           labels.astype('int32'),
           weights.astype('float32')])
