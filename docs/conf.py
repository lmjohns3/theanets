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
    'sphinxcontrib.tikz',
    'numpydoc',
    ]
autosummary_generate = True
autodoc_default_flags = ['members']
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
templates_path = ['_templates']
source_suffix = '.rst'
source_encoding = 'utf-8-sig'
master_doc = 'index'
project = u'theanets'
copyright = u'2014, Leif Johnson'
version = '0.4'
release = '0.4.0'
exclude_patterns = ['_build']
pygments_style = 'tango'

html_theme = 'readable'
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
html_context = dict(google_analytics_id='UA-57658-10')
htmlhelp_basename = 'theanetsdoc'

latex_elements = {
#'papersize': 'letterpaper',
#'pointsize': '10pt',
'preamble': r'''
\usepackage{tikz}
\usepackage{pgfplots}
\usetikzlibrary{arrows}''',
}
latex_documents = [
  ('index', 'theanets.tex', u'theanets Documentation',
   u'Leif Johnson', 'manual'),
]

man_pages = [
    ('index', 'theanets', u'theanets Documentation',
     [u'Leif Johnson'], 1)
]

texinfo_documents = [
  ('index',
   'theanets',
   u'theanets Documentation',
   u'Leif Johnson',
   'theanets',
   'One line description of project.',
   'Miscellaneous'),
]

intersphinx_mapping = {'http://docs.python.org/': None}
