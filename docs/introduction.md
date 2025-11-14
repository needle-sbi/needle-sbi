# Introduction

The orchestrator module pieces together the data-management objects from the preprocessor submodule
and the machine learning tools from the ml submodule. It makes use of the following libraries:

 - law (luigi fork) for task scheduling and remote job submissions
 - lightning for interfacing with lightning modules in the ml submodule
 - hydra, omegaconf for managing configs

## Structure

We want to keep the pure law orchestration separate from the ml tools that are run inside of them.
Apart from the law Tasks themselves, the orchestration module is lightweight. It includes an
`orchestration` folder with some utils, that could also be renamed to utils.

```
law_tasks
 -> mixins
orchestrator
ml              # submodules
preprocessor    # submodules
``` 
