# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

#sys.path.insert(0, os.path.abspath('../..'))

project = 'spring'
copyright = '2024, Alessandro Colombo'
author = 'Alessandro Colombo'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','sphinx_autodoc_typehints',"myst_nb"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_static_path = ['_static']
#import sphinx_adc_theme
#html_theme = 'sphinx_adc_theme'
#html_theme_path = [sphinx_adc_theme.get_html_theme_path()]

html_theme = 'sphinx_rtd_theme'
html_logo = "figures/springlogo_small.png"
nb_execution_mode = "off"
