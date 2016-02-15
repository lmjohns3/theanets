import numpy as np
import theanets
import theanets.util


class TestRandomMatrix:
    def test_sparsity(self):
        x = theanets.util.random_matrix(1000, 200, sparsity=0.1, rng=5)
        assert x.shape == (1000, 200)
        assert np.allclose(x.mean(), 0, atol=1e-2), x.mean()
        assert np.allclose(x.std(), 0.95, atol=1e-2), x.std()
        assert np.allclose((x == 0).mean(), 0.1, atol=1e-1), (x == 0).mean()

    def test_diagonal(self):
        x = theanets.util.random_matrix(1000, 200, diagonal=0.9, rng=4)
        assert x.shape == (1000, 200)
        assert np.allclose(np.diag(x), 0.9), np.diag(x)
        assert x.sum() == 180, x.sum()

    def test_radius(self):
        x = theanets.util.random_matrix(1000, 200, radius=2, rng=4)
        assert x.shape == (1000, 200)
        u, s, vT = np.linalg.svd(x)
        assert s[0] == 2, s
        assert s[1] < 2


class TestRandomVector:
    def test_rng(self):
        x = theanets.util.random_vector(10000, rng=4)
        assert x.shape == (10000, )
        assert np.allclose(x.mean(), 0, atol=1e-2)
        assert np.allclose(x.std(), 1, atol=1e-2)


class TestMatching:
    def test_params_matching(self):
        net = theanets.Autoencoder([10, 20, 30, 10])

        match = sorted(theanets.util.params_matching(net.layers, '*'))
        assert len(match) == 6
        assert [n for n, _ in match] == [
            'hid1.b', 'hid1.w', 'hid2.b', 'hid2.w', 'out.b', 'out.w']

        match = sorted(theanets.util.params_matching(net.layers, '*.w'))
        assert len(match) == 3
        assert [n for n, _ in match] == ['hid1.w', 'hid2.w', 'out.w']

        match = sorted(theanets.util.params_matching(net.layers, 'o*.?'))
        assert len(match) == 2
        assert [n for n, _ in match] == ['out.b', 'out.w']

    def test_outputs_matching(self):
        outputs, _ = theanets.Autoencoder([10, 20, 30, 10]).build_graph()

        match = sorted(theanets.util.outputs_matching(outputs, '*'))
        assert len(match) == 7
        assert [n for n, _ in match] == [
            'hid1:out', 'hid1:pre', 'hid2:out', 'hid2:pre',
            'in:out', 'out:out', 'out:pre']

        match = sorted(theanets.util.outputs_matching(outputs, 'hid?:*'))
        assert len(match) == 4
        assert [n for n, _ in match] == [
            'hid1:out', 'hid1:pre', 'hid2:out', 'hid2:pre']

        match = sorted(theanets.util.outputs_matching(outputs, '*:pre'))
        assert len(match) == 3
        assert [n for n, _ in match] == ['hid1:pre', 'hid2:pre', 'out:pre']
