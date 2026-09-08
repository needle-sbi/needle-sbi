### Markers

- `@pytest.mark.law` This marker requires the usage of a temporary directory in order to not clash with existing or missing dependencies. The best way to approach this is to use the built-in `tmp_path` pytest fixture and set the `results_dir_path` parameter to that Path.
- `@pytest.mark.slow` Flags a test as slow (> 1sec). For regular testing these are skipped, but for benchmarks one must explicitly set `-m "not slow"` in the CLI to avoid these long tests.


### Benchmarks

 - `pytest --benchmark-only` Will run all benchmark-related tests. Most will require environment variables or links to the corresponding datasets in the config. Run preferably on powerful hardware.
 - `pytest --benchmark-only --benchmark-plot` for an extra plotting script showing the root vs. parquet ingestion speed.

