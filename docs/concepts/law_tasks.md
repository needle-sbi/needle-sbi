
# LAW Tasks

Law is a fork of the luigi project by Spotify that applies python task scheduling to HEP environments.

 - **Luigi** [docs](https://luigi.readthedocs.io/en/stable/)
 - **Law** [docs](https://law.readthedocs.io/en/latest/).
 - **B2luigi** [docs](https://b2luigi.belle2.org/index.html).
    This is another fork of luigi by the Belle II
    collaboration, which in my opinion is much nicer than law. However, they have less features and
    is not used outside of the Belle II collaboration. It also has very nice documentation and the
    main developers are from KIT.
 - **ColumnFlow** [docs](https://columnflow.readthedocs.io/en/latest/).
    Provides a bit of context as to how to use Law, even though most Tasks in ColumnFlow are already
    implemented.
---
## What is an orchestration tool?

Most HEP analysis are chunked into jobs: independent chunks of code and data that can run in an isolated
environment on a batch system. There are several types of schedulers that determine in what order jobs
are executed, for example *Slurm* or *HTCondor*. The old-schooled way of interfacing with a scheduler is
to write a Job Description Language (.jdl) file. These list the requirements of a given task in terms
of memory or number of cores or even what container to run the code in.

Orchestration tools such as luigi or Law remove the overhead of writing these `.jdl` files manually. You
write python classes called Tasks that implement methods describing how to run you code. In addition,
the requirements of each Task can be added as an attribute to your class. This makes it very easy to
write modular workflow in a Directed Acyclic Graph (DAG). These workflows start with a WrapperTask (also
called the main Task), which lists a set of required Tasks that have to be run before that Task is
marked as complete.

A Task is counted as complete if:
 1. All its requirements are complete. So all the Tasks that this Task depends on are marked as complete.
    This implies a recursive check that works up the DAG until it finds a Task that is not yet complete.
 2. All its outputs exist. Output files or folders are defined using the `output()` method. The outputs
    have to be created during the execution of the Task.

The `run()` method is responsible for actually executing the main body of code that the Task is supposed
to perform. If after reaching the end of the `run()` block an output file is missing, the Task is marked
as failed and the whole DAG might stop. This is intended behavior since otherwise downstream tasks will
fail due to inexistent files that they in turn depend on.

A Task might require:
 1. Other Tasks using the `req()` method in law, or `requires()` in luigi. These are just other Tasks
    with associated parameters.
 2. Input files. In Law, `input()` provides a way to access the outputs of the required Tasks for this
    Task to use. Basically, you define `output()` with each output file having a fixed name and you
    access these names in the next Task using the `input()` method with that same name. If that file
    does not exist or the name is wrong, Law will raise an `Unfulfilled dependencies at RunTime` Error
    and tell you which files it expected.

## Simple example

Let us take a simple example where I want to run 5 different k-fold trainings. First, I define the "leaf"
Task for each Fold:

```python
import os

import law
import luigi


class FoldTask(law.Task):  # inherit from the law.Task
    fold_index = luigi.IntParameter(  # CLI parameter
        description="K-Fold index",
        significant=True,  # whether the name of Task should include fold_index=...
    )

    def output(self):  # list what files should exist at the end of this Task
        return {
            "outputs": law.LocalFileTarget(os.path.abspath(f"fold_{self.fold_index}/training_output.json")),
        }

    def run(self):
        # some code that runs my training ...

        results = ...
        self.output()["outputs"].touch()  # create the directory automatically
        results.to_json(self.output()["outputs"].path)  # save the results where law expects them
```

There are some details to explain:
 - There is no fixed format for your outputs, these just have to be an Iterable of paths that law can
    check the existence of. The paths must however be absolute paths. Relative paths will be declared
    as missing or cannot be found.
 - Your Task might run successfully if it finishes its `run()` method, but it could still be that the
    outputs are missing. Usually this is not the case but for debugging purposes you should check where
    the files were saved if it happened at all.
 - **Parameter** are command line arguments that can be set by you using the `--fold_index 1` flag for
    example (with the name of the parameter). You can override them in python of course. You can also
    set a default value or store the default in the `law.cfg` file. Lot of options here.

More info can be found in the relevant docs for law or luigi.

## Example with requirements

Lets go for a Task that requires five copies of the previous FoldTask each with a different law Parameter.
In NEEDLE, `EnsembleTask` reads the number of folds from the Hydra config via the `HydraMixin`:

```python
"""
Task for a single ensemble training, includes multiple folds.
"""
import os
from pathlib import Path
from typing import Any, Dict

import law
import luigi

from law_tasks.fold import FoldTask
from law_tasks.mixins import HydraMixin
from needle.utils.config_schema import EstimatorConfig
from needle.utils.results import FoldResults, EnsembleResults  # dataclasses with JSON serialization


class EnsembleTask(HydraMixin, law.Task):
    results_path: str = law.Parameter(
        description="Root directory where results are saved.",
        significant=False,
    )
    estimator: str = law.Parameter(
        description="Name of the estimator (must be included in config).",
        significant=True,
    )
    systematic: str = law.Parameter(
        description="Name of the systematic uncertainty.",
        significant=True,
    )
    ensemble: int = luigi.IntParameter(default=0, significant=True)

    @property
    def estimator_config(self) -> EstimatorConfig:
        return self.config.estimators[self.estimator]

    def requires(self):
        return [
            FoldTask(
                config_file=self.config_file,
                hydra_overrides=self.hydra_overrides,
                estimator=self.estimator,
                systematic=self.systematic,
                ensemble=self.ensemble,
                fold_index=fold_index,
                results_path=self.results_path,
            )
            for fold_index in range(self.estimator_config.expands.folds)
        ]  # n_folds is read from config.estimators[estimator].expands.folds

    def output(self) -> Dict[str, Any]:
        base = law.LocalDirectoryTarget(os.path.abspath(self.results_path))
        return {"outputs": base.child("ensemble_results.json", type="f")}

    def run(self):
        fold_results = [
            FoldResults.from_json(fold_output["outputs"].path)  # load each fold's output
            for fold_output in self.input()
        ]
        EnsembleResults(folds=fold_results).to_json(self.output()["outputs"].path)
```

Key points:
 - **`HydraMixin` must come before `law.Task`** in the MRO. It adds the `config_file` and
    `hydra_overrides` parameters plus the `config` property that lazily loads and caches the
    parsed Hydra config.
 - The number of folds is not a hard-coded parameter but is read from
    `self.estimator_config.expands.folds`, which is resolved from the YAML config at runtime.
 - All parameters that identify the parent task (`config_file`, `hydra_overrides`, `estimator`,
    `systematic`, `ensemble`) are forwarded explicitly when constructing `FoldTask` instances.

## More on Law

You can customize a lot how the Tasks are to be run. Adding the `law.htcondor.HTCondorWorkflow` Mixin
allows you to run that Task on a batch system automatically. You still have to configure some HTCondor
stuff but you can define that per Task, depending on your requirements.

For a complete working example of this pattern in a real HEP analysis pipeline, see the
[FAIR Universe demo](../examples/index.md).
