# Orchestrator for NEEDLE ML workflows

## Installation

 1. Clone the repository along with its two submodules [ml](https://gitlab.desy.de/needle/ml) and [preprocessor](https://gitlab.desy.de/needle/preprocessor) using

    ```bash
    git clone git@gitlab.desy.de:needle/orchestrator.git --recurse-submodules
    ```

 2. Register the submodules with
    
    ```bash
    git submodule update --init --recursive
    ```

## Setup

 1. Installing the required python libraries
      - **Option A**:
         
         Create the python environment using the `requirements.txt` file. This gives you full control over your libraries

      - **Option B**:
         
         Run your code inside our `needle.sif` singularity container image. The required libraries are already installed in the container.
         
         To create the container from scratch, use the `singularity_dev.def` (with all files copied) or `singularity_base.df` (with only the dependencies installed). The def files are found in the container folder.

         Running the container:

         - If you are only using `singularity run`, you may skip step 2, as that is performed automatically within the runscript of the container.
          - For `singularity exec` and `singularity shell`, you still need to set up the environment using step 2.

 2. Setting up LAW:

    In order to use the [law](https://github.com/riga/law) workflow manager, run

    ```bash
    source setup.sh  # sets up environment variables
    law index  # looks for all available LAW tasks
    ```
 3. Running LAW tasks:

    This will allow you to run the workflow my calling the corresponding `law` Task from the command line with

    ```bash
    law run tasks.<LawTask>  # if not specified 'tasks' in law.cfg
    law run TrainingBaseTask  # this default Task is already in law.cfg
    ``` 

    More information can be found on the `law` [documentation](https://law.readthedocs.io/en/latest/)

## Pushing with submodules

There are two submodules registered in the orchestrator repository. To automatically push changes made to these submodules when you push to orchestrator, use

```bash
git push --recurse-submodules=on-demand
```

## Tensorboard

All log files are saved to TensorBoard under `runs/tensorboard_logs` (which cannot be changed in the config for now). The file paths are managed by LAW. To access them, either open the logs using the browser or with VSCode by installing the Tensorboard Extension and then using CMD+SHIFT+P to open the command palette and selecting "Python: Launch Tensorboard". Once prompted, manually set the log directory to `runs/tensorboard_logs`. A new tab with Tensorboard will then open.
