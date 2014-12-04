import numpy as np
import theanets


class TestDataset:
    def setUp(self):
        np.random.seed(3)

    def test_iterate(self):
        dataset = theanets.dataset.Dataset(
            np.random.randn(100, 2), iteration_size=4, batch_size=10)

        assert len(dataset.batches) == 10

        batches_ = list(dataset.batches)

        # check we iterate over the correct number of batches.
        assert sum(1 for _ in dataset) == 4

        # check the dataset didn't get shuffled (yet).
        assert all(a is b for a, b in zip(dataset.batches, batches_))

        assert sum(1 for _ in dataset) == 4
        assert sum(1 for _ in dataset) == 4

        # check the dataset did get shuffled.
        assert not all(a is b for a, b in zip(dataset.batches, batches_))

    def test_threedee(self):
        dataset = theanets.dataset.Dataset(
            np.random.randn(5, 102, 3), iteration_size=13, batch_size=10)

        i = 0
        for x, in dataset:
            print(x.shape)
            assert x.shape == (5, 10, 3) or x.shape == (5, 2, 3)
            i += 1
        assert i == 13
