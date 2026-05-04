from __future__ import annotations

import importlib.metadata

project = "NEEDLE"
copyright = "2026, Needle Team"

# resolve version from the installed distribution, or fall back
try:
    version = release = importlib.metadata.version("needle-orchestrator")
except importlib.metadata.PackageNotFoundError:
    version = release = "0+local"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "NEEDLE",
    },
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitLab",
            "url": "",  # TODO: update with real URL (deploy docs)
            "icon": "fa-brands fa-gitlab",
        },
    ],
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_depth": 4,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
}

templates_path = ["_templates"]

html_sidebars = {
    "**": ["sidebar-nav-all.html"],
}

html_title = f"{project} v{version}"
html_short_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_last_updated_fmt = "%b %d, %Y"

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

myst_dmath_double_inline = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "awkward": ("https://awkward-array.org/doc/main/", None),
}

# mock heavy/optional deps so autodoc doesn't need them installed
autodoc_mock_imports = [
    "torch",
    "lightning",
    "pytorch_lightning",
    "mlflow",
    "hydra",
    "omegaconf",
    "law",
    "luigi",
    "dask",
    "dask_awkward",
    "uproot",
    "awkward",
    "tensorboard",
    "pydantic",
    "psutil",
    "pyarrow",
    "spacy",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}

autodoc_typehints = "both"

nitpick_ignore = [
    ("py:class", "SerializableDataclass"),
    ("py:class", "law.Task"),
    ("py:class", "law.Parameter"),
    ("py:class", "luigi.IntParameter"),
    ("py:class", "L.LightningModule"),
    ("py:class", "L.LightningDataModule"),
]
