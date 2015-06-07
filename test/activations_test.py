import theanets


class TestBuild:
    def test_build(self):
        a = theanets.layers.Feedforward(inputs=2, size=3, activation='relu').activate
        assert callable(a)
        assert a.name == 'relu'
        assert a.params == []

    def test_build_composed(self):
        a = theanets.layers.Feedforward(inputs=2, size=3, activation='relu+norm:z').activate
        assert callable(a)
        assert a.name == 'norm:z(relu)', a.name
        assert a.params == []


class TestParametric:
    def test_prelu(self):
        l = theanets.layers.Feedforward(inputs=2, size=3, activation='prelu', name='l')
        actual = sorted(p.name for p in l.params)
        assert actual == ['l.b', 'l.leak', 'l.w'], actual

    def test_lgrelu(self):
        l = theanets.layers.Feedforward(inputs=2, size=3, activation='lgrelu', name='l')
        actual = sorted(p.name for p in l.params)
        assert actual == ['l.b', 'l.gain', 'l.leak', 'l.w'], actual

    def test_maxout(self):
        l = theanets.layers.Feedforward(inputs=2, size=3, activation='maxout:8', name='l')
        actual = sorted(p.name for p in l.params)
        assert actual == ['l.b', 'l.intercept', 'l.slope', 'l.w'], actual
