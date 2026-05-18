
## Locally build the documentation

To build the documentation, follow these steps:

```bash
# install docs dependencies from pyproject [dependency-groups].docs
uv sync --group docs
```

then run the following to build the docs:

```bash
uv run python -m sphinx -T -b html -d docs/_build/doctrees -D language=en docs docs/_build/html
```

You can view the documentation locally by then running the following:

```bash
open docs/_build/html/index.html
```

If the code is on a remote machine, forward a remote port to your local machine, for example with ssh:

```bash
ssh -L 8801:localhost:8801 <user>@<remote-server>  # on local
```

Then create a remote python http server using:

```bash
python3 -m http.server 8801 --bind 127.0.0.1  # on remote
```

Which allows you to look at the docs folder on http://127.0.0.1:8801/.
