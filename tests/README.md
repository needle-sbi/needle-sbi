### Markers

- `@pytest.mark.law` This marker requires the usage of a temporary directory in order to not clash with existing or missing dependencies. The best way to approach this is to use the built-in `tmp_path` pytest fixture and set the `results_dir_path` parameter to that Path.
