### Markers

- `@pytest.mark.law` This marker requires the usage of a temporary directory in order to not clash with existing or missing dependencies. The best way to approach this is to use the built-in `tmp_path` pytest fixture and set the `results_dir_path` parameter to that Path.


### Benchmarks

 - `pytest --benchmark-only` Will run all benchmark-related tests. Most will require environment variables or links to the corresponding datasets in the config. Run preferably on powerful hardware.
