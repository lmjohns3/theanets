import theanets


class TestBuild:
    def test_build_mse(self):
        l = theanets.losses.build('mse', in_dim=2, out_dim=1)
        assert callable(l)
        assert len(l.variables) == 2

    def test_build_mae(self):
        l = theanets.losses.build('mae', in_dim=2)
        assert callable(l)
        assert len(l.variables) == 1

    def test_build_mae_weighted(self):
        l = theanets.losses.build('mae', in_dim=2, weighted=True)
        assert callable(l)
        assert len(l.variables) == 2

    def test_kl(self):
        l = theanets.losses.build('kl', in_dim=2, out_dim=2)
        assert callable(l)
        assert len(l.variables) == 2
