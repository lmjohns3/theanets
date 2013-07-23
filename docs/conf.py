import sys, os
#sys.path.insert(0, os.path.abspath('.'))

#needs_sphinx = '1.0'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    ]
templates_path = ['_templates']
source_suffix = '.rst'
#source_encoding = 'utf-8-sig'
master_doc = 'index'
project = u'theanets'
copyright = u'2013, Leif Johnson'
version = '0.1.0'
release = '0.1.0'
#language = None
#today = ''
#today_fmt = '%B %d, %Y'
exclude_patterns = ['_build']
#default_role = None
#add_function_parentheses = True
#add_module_names = True
#show_authors = False
pygments_style = 'sphinx'
#modindex_common_prefix = []

html_theme = 'nature'
#html_theme_options = {}
#html_theme_path = []
#html_title = None
#html_short_title = None
#html_logo = None
#html_favicon = None
html_static_path = ['_static']
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
#'preamble': '',
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
