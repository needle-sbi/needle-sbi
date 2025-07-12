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

 1. Create the python environment using the `requirements.txt` file. This is an early development stage, therefore we do not ship a finished environment yet
 2. In order to use the [law](https://github.com/riga/law) workflow manager, run

    ```bash
    law index
    ```

    This will allow you to run the workflow my calling the corresponding `law` Task from the command line with

    ```bash
    law run tasks.<LawTask>
    ``` 

    More information can be found on the `law` [documentation](https://law.readthedocs.io/en/latest/)

## Tensorboard

All log files are saved to TensorBoard under `runs/tensorboard_logs` (which cannot be changed in the config for now). The file paths are managed by LAW. To access them, either open the logs using the browser or with VSCode by installing the Tensorboard Extension and then using CMD+SHIFT+P to open the command palette and selecting "Python: Launch Tensorboard". Once prompted, manually set the log directory to `runs/tensorboard_logs`. A new tab with Tensorboard will then open.
