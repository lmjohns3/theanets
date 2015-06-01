import glob
import os
import pep8


class TestCodeFormat:
    def test_pep8(self):
        def match(*p):
            s = list(p) + ['*.py']
            return glob.glob(os.path.join(*s))
        pep8style = pep8.StyleGuide(config_file='setup.cfg')
        result = pep8style.check_files(
            match('examples') +
            match('test') +
            match('theanets') +
            match('theanets', 'layers'))
        assert result.total_errors == 0
