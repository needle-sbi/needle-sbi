# Setup

This page covers everything you need to go from a fresh checkout to a running training. For an
overview of the different ways to actually run tasks once set up, see [Usage](usage.md).

## Prerequisites

You need:
- Python 3.12+
- A virtual environment (`.venv/` at the repo root, created e.g. with `python -m venv .venv`). The
    name of the environment is not relevant, but we will use `.venv` as a convention. You can also use
    any other environment manager like `conda`, as long as it is compatible with `pyproject.toml`.
- (Optional) The FAIR Universe dataset if running that example (see `FAIR_UNIVERSE_DATA` below)

## First time package installation

### Option 1: Plain pip

Create or use an existing virtual environment with `python3 -m venv`. Install the `needle` package with

```bash
pip install "git+ssh://git@github.com/needle-sbi/needle-sbi.git"
```

### Option 2: Astral uv

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install `uv` with

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload your shell or source the uv environment to make it active in your current shell, as stated by
uv when you run the script.

```bash
uv pip install --no-config "git+ssh://git@github.com/needle-sbi/needle-sbi.git"
```

### Initialize the workspace

In order to run `needle` you need a few config files. These can be created automatically using the
CLI tool. In your virtual environment run:

```bash
needle init
```

This will copy:

- `setup.sh`: script for activating the NEEDLE environment
- `conf/`: template directory for your config files according to the hydra schema
- `law.cfg`: law config file for batch submissions
- `settings.json`: b2luigi config file for batch submissions
- `index`: law index file that lists all the available tasks. Can be updated using `law index`

You can also use `needle init --backend law` or `needle init --backend b2luigi` to only
initialize the project with the backend that you want. Otherwise you keep both options open.

### Set up the NEEDLE environment

```bash
source setup.sh
```

::: {important}
Always source your virtual environment and then run the `setup.sh` script every time you open a new
shell.
:::

::: {admonition} Why `source setup.sh` and not just `python -m law`?
:class: note
LAW needs to know which Python modules contain your tasks. The `setup.sh` script sets the `LAW_HOME`
and `LAW_CONFIG_FILE` environment variables so LAW can pick up the `law.cfg` file automatically.
:::

## Quick test

Try out an absolutely minimal example with the default config (with a mock transformer model) using

```bash
needle run
```

Once this works, head over to [Usage](usage.md) to learn about the different ways to run tasks.
