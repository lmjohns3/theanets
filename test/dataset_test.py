import numpy as np
import theanets


class TestDataset:
    def test_iterate(self):
        dataset = theanets.dataset.Dataset(
            np.random.randn(101, 1),
            label='foo',
            batches=4,
            size=10,
        )

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
            np.random.randn(101, 202, 3),
            batches=4,
            size=20,
        )

        i = 0
        for x, in dataset:
            assert x.shape == (101, 20, 3)
            i += 1
        assert i == 4
