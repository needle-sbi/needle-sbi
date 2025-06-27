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
