Graph-level representations using ensemble-based readout functions
====
The following is repository containing implementation of the paper _"Graph-level representations using ensemble-based readout functions"_
by Jakub Binkowski, Albert Sawczyn, Denis Janiak, Piotr Bielak and Tomasz Kajdanowicz.

# Experiments

## Remarks
1. Experiment are stated in `dvc.yaml` file, and could be easly run with `dvc repro`  command (but not in parallel, so it may take much time to finish)
2. Repeated experiments on several seeds are run with script: [train_gnn_with_reps.py](experiments%2Fscripts%2Ftrain_gnn_with_reps.py)
3. Each experiment runs with corresponding configuration file `experiments/confg/ensemble_readouts/hparams_<dataset>.yaml`, which contains:
   - directories of the source dataset and further experiment output
   - hyperparameters, including model architecture specification

## Installation
Repository relies on **Python 3.10**

- There are 4 files with requirements declared:
  - `requirements-cpu.txt` - packages only for CPU environments
  - `requirements-gpu.txt` - packages only for GPU environments
  - `requirements.txt` - packages common to GPU and CPU environments
  - `requirements-dev.txt` - linting tool to support code quality maintenance (optional)
   
- Install requirements for CPU environments:
  ```shell
   pip install -r requirements-cpu.txt -r requirements.txt -r requirements-dev.txt
  ```
- Install requirements for GPU environments:
   ```shell
   pip install -r requirements-gpu.txt -r requirements.txt -r requirements-dev.txt
  ```
## Reproducing with DVC
To reproduce experiments with DVC, simply use the command below. Due to relatively low resource consumption,
one might want to leverage parallel run described in the next section.
```shell
$ dvc pull data/datasets/{ENZYMES,MUTAG,REDDIT-MULTI-12K,ZINC}.dvc
$ dvc repro
```

## Running parallel experiments
- Single experiment is relatively lightweight (consumes about ~10% of NVIDIA TITAN RTX)
- For the sake of fast computation of multiple experiments there is script, which helps to exploit resources
- To run experiments:
  - Ensure you have conda with environment called `ensemble-readouts`, containing all dependencies installed
  - Select dataset for which you want to run experiments, and look at the script [run_training.sh](experiments%2Fshell_runner%2Frun_training.sh)
  - Run the scripts with parameters depending on available resources, e.g.,
    ```shell
    CUDA_VISIBLE_DEVICES=0 experiments/shell_runner/run_training.sh \
      --config-path experiments/config/deterministic_readouts/hparams_enzymes.yaml \
      --config-list experiments/config/config_names.yaml \
      --config-list-names ensemble_readouts \
      --num-jobs 4 \
      --accelerator gpu \
      --devices 1
    ```
