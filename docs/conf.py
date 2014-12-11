import os
import sys

import sphinx_rtd_theme

if os.environ.get('READTHEDOCS', None) == 'True':
    os.environ['PATH'] += os.pathsep + os.path.abspath('_bin')
    os.environ['LD_LIBRARY_PATH'] = os.path.abspath('_bin')

#needs_sphinx = '1.0'
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
templates_path = ['_templates']
source_suffix = '.rst'
source_encoding = 'utf-8-sig'
master_doc = 'index'
project = u'theanets'
copyright = u'2014, Leif Johnson'
version = '0.2.0'
release = '0.2.0'
#language = None
#today = ''
#today_fmt = '%B %d, %Y'
exclude_patterns = ['_build']
#default_role = None
#add_function_parentheses = True
#add_module_names = True
#show_authors = False
pygments_style = 'tango'
#modindex_common_prefix = []

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']
html_context = dict(
    google_analytics_id='UA-57658-4',
)
#html_title = None
#html_short_title = None
#html_logo = None
#html_favicon = None
#html_last_updated_fmt = '%b %d, %Y'
#html_use_smartypants = True
#html_sidebars = {}
#html_additional_pages = {}
#html_domain_indices = True
#html_use_index = True
#html_split_index = False
#html_show_sourcelink = True
#html_show_sphinx = True
#html_show_copyright = True
#html_use_opensearch = ''
#html_file_suffix = None
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
#latex_logo = None
#latex_use_parts = False
#latex_show_pagerefs = False
#latex_show_urls = False
#latex_appendices = []
#latex_domain_indices = True

man_pages = [
    ('index', 'theanets', u'theanets Documentation',
     [u'Leif Johnson'], 1)
]
#man_show_urls = False

texinfo_documents = [
  ('index',
   'theanets',
   u'theanets Documentation',
   u'Leif Johnson',
   'theanets',
   'One line description of project.',
   'Miscellaneous'),
]
#texinfo_appendices = []
#texinfo_domain_indices = True
#texinfo_show_urls = 'footnote'

intersphinx_mapping = {'http://docs.python.org/': None}
