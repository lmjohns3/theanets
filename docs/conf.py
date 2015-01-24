import os
import sys

import sphinx_readable_theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    #'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    #'sphinx.ext.pngmath',
    'sphinx.ext.viewcode',
    'numpydoc',
    ]
autosummary_generate = True
autodoc_default_flags = ['members']
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = True
source_suffix = '.rst'
source_encoding = 'utf-8-sig'
master_doc = 'index'
project = u'theanets'
copyright = u'2014, Leif Johnson'
version = '0.4'
release = '0.4.1'
exclude_patterns = ['_build']
templates_path = ['_templates']
pygments_style = 'tango'

html_theme = 'readable'
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
htmlhelp_basename = 'theanetsdoc'

def h(*xs):
    return ['{}.html'.format(x) for x in xs]
html_sidebars = {
    'index': h('gitwidgets', 'globaltoc', 'searchbox'),
    '**': h('gitwidgets', 'localtoc', 'relations', 'sourcelink', 'searchbox'),
}

intersphinx_mapping = {
    'python': ('http://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
}
