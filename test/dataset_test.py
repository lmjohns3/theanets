import numpy as np
import theanets


class TestDataset:
    def setUp(self):
        self.dataset = theanets.dataset.Dataset(
            np.arange(101)[:, None],
            label='foo',
            batches=4,
            size=10,
        )

    def test_setup(self):
        assert self.dataset.label == 'foo'
        assert len(self.dataset.batches) == 10
        assert self.dataset.number_batches == 4

    def test_iterate(self):
        batches_ = list(self.dataset.batches)

        # check we iterate over the correct number of batches.
        assert sum(1 for _ in self.dataset) == 4

        # check the dataset didn't get shuffled (yet).
        assert all(a is b for a, b in zip(self.dataset.batches, batches_))

        assert sum(1 for _ in self.dataset) == 4
        assert sum(1 for _ in self.dataset) == 4

        # check the dataset did get shuffled.
        assert not all(a is b for a, b in zip(self.dataset.batches, batches_))
