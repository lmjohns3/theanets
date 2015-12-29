import theanets

import util


class Mixin:
    def assert_progress(self, **kwargs):
        start = best = None
        for _, val in self.exp.itertrain(
                [self.INPUTS, self.OUTPUTS],
                algorithm='sgd',
                patience=2,
                min_improvement=0.01,
                batch_size=self.NUM_EXAMPLES,
                **kwargs):
            if start is None:
                start = best = val['loss']
            if val['loss'] < best:
                best = val['loss']
        assert best < start   # should have made progress!


class TestNetwork(Mixin, util.Base):
    def setUp(self):
        self.exp = theanets.Regressor(
            [self.NUM_INPUTS, 20, self.NUM_OUTPUTS], rng=131)

    def test_input_noise(self):
        self.assert_progress(input_noise=0.001)

    def test_input_dropout(self):
        self.assert_progress(input_dropout=0.1)

    def test_hidden_noise(self):
        self.assert_progress(hidden_noise=0.001)

    def test_hidden_dropout(self):
        self.assert_progress(hidden_dropout=0.1)

    def test_noise(self):
        self.assert_progress(noise={'*:out': 0.001})

    def test_dropout(self):
        self.assert_progress(dropout={'*:out': 0.1})

    def test_hidden_l1(self):
        self.assert_progress(hidden_l1=0.001)

    def test_weight_l1(self):
        self.assert_progress(weight_l1=0.001)

    def test_weight_l2(self):
        self.assert_progress(weight_l2=0.001)

    def test_contractive(self):
        self.assert_progress(contractive=0.001)


class TestRecurrent(Mixin, util.RecurrentBase):
    def setUp(self):
        self.exp = theanets.recurrent.Regressor([
            self.NUM_INPUTS, (20, 'rnn'), self.NUM_OUTPUTS])

    def test_recurrent_matching(self):
        regs = theanets.regularizers.from_kwargs(self.exp)
        outputs, _ = self.exp.build_graph(regs)
        matches = theanets.util.outputs_matching(outputs, '*:out')
        hiddens = [(n, e) for n, e in matches if e.ndim == 3]
        assert len(hiddens) == 3, [n for n, e in hiddens]

    def test_recurrent_norm(self):
        self.assert_progress(recurrent_norm=0.001)
