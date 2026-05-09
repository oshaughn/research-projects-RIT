# Configuration file for the SpAhinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# Add RIFT module path for autodoc
sys.path.insert(0, os.path.abspath('../../MonteCarloMarginalizeCode/Code'))
# Add current source path to allow importing md_converter
sys.path.insert(0, os.path.abspath('.'))

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'RIFT'
copyright = '2022, K Wagner'
author = 'K Wagner'

# The full version, including alpha/beta/rc tags
release = '"2022, Richard O\'Shaughnessy et al"'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

master_doc = "index"

extensions = ["sphinx.ext.autodoc",
              "sphinx_rtd_theme",
              #"sphinx_tabs.tabs",
              "sphinx_multiversion",
              #"sphinx_toolbox.collapse"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', "requirements.txt"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "canonical_url": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

import md_converter

def setup(app):
    # Dynamically convert DESIGN.md to RST before the build
    source_dir = os.path.dirname(__file__)
    md_file = os.path.abspath(os.path.join(source_dir, '../../MonteCarloMarginalizeCode/Code/RIFT/simulation_manager/DESIGN.md'))
    rst_file = os.path.abspath(os.path.join(source_dir, 'api_reference/simulation_manager/design_overview.rst'))
    
    if os.path.exists(md_file):
        # Prefer pandoc if available for a high‑quality conversion
        try:
            import subprocess, shlex
            subprocess.run(shlex.split(f"pandoc {md_file} -f markdown -t rst -o {rst_file}"), check=True)
        except Exception:
            # Fallback to the simple Python converter
            md_converter.convert_md_to_rst(md_file, rst_file)

