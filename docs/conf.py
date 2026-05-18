from __future__ import annotations

import importlib.metadata

project = "NEEDLE"
copyright = "2026, Needle Team"

# resolve version from the installed distribution, or fall back
try:
    version = release = importlib.metadata.version("needle-sbi")
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
    "networkx",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_typehints = "both"

maximum_signature_line_length = 1

nitpick_ignore = [
    ("py:class", "SerializableDataclass"),
    ("py:class", "law.Task"),
    ("py:class", "law.Parameter"),
    ("py:class", "luigi.IntParameter"),
    ("py:class", "L.LightningModule"),
    ("py:class", "L.LightningDataModule"),
]


def _strip_config_schema_params(app, what, name, obj, options, lines):
    """Remove injected :param:/:type: lines for config_schema dataclasses.

    autodoc_typehints="both" injects these into the class description, producing a
    "Parameters:" section that duplicates the signature already shown at the top.
    """
    if what != "class" or not name.startswith("needle.utils.config_schema."):
        return
    lines[:] = [line for line in lines if not line.startswith(":param ") and not line.startswith(":type ")]


def setup(app: object) -> None:
    # sphinx's _MockObject does not define __add__ or __or__, which breaks
    # class-body code like `interactive_params = law.Task.interactive_params + [...]`
    # and overloaded methods with `dak.Array | ak.Array` annotations.
    from sphinx.ext.autodoc.mock import _MockObject

    _MockObject.__add__ = lambda self, other: other if isinstance(other, list) else _MockObject()  # type: ignore[attr-defined]
    _MockObject.__radd__ = lambda self, other: other if isinstance(other, list) else _MockObject()  # type: ignore[attr-defined]
    _MockObject.__or__ = lambda self, other: _MockObject()  # type: ignore[attr-defined]
    _MockObject.__ror__ = lambda self, other: _MockObject()  # type: ignore[attr-defined]

    app.connect("autodoc-process-docstring", _strip_config_schema_params)
