# FJSP-DRL

This repository is the official implementation of the paper “Flexible Job Shop Scheduling via Dual Attention Network Based Reinforcement Learning”

## Quick Start

### requirements

- python $=$ 3.7.11
- numpy $=$ 1.21.6
- ortools $=$ 9.3.10497
- pandas $=$ 1.3.5
- torch $=$ 1.11.0+cu113
- torchaudio $=$ 0.11.0+cu113
- torchvision $=$ 0.12.0+cu113
- tqdm $=$ 4.64.0

### introduction

- ``
- 
- `data_dev` and `data_test` are the validation sets and test sets, respectively.
- `data` saves the instance files generated by `./utils/create_ins.py`
- `env` contains code for the DRL environment
- `graph` is part of the code related to the graph neural network
- `model` saves the model for testing
- `results` saves the trained models
- `save` is the folder where the experimental results are saved
- `utils` contains some helper functions
- `config.json` is the configuration file
- `mlp.py` is the MLP code (referenced from L2D)
- `PPO_model.py` contains the implementation of the algorithms in this article, including HGNN and PPO algorithms
- `test.py` for testing
- `train.py` for training
- `validate.py` is used for validation without manual calls



### train



### evaluate

There are various experiments in this article, which are difficult to be covered in a single run. Therefore, please change `config.json` before running.

Note that disabling the `validate_gantt()` function in `schedule()` can improve the efficiency of the program, which is used to check whether the solution is feasible.

### train

```
python train.py
```



Note that there should be a validation set of the corresponding size in `./data_dev`.

### test

```
python test.py
```



Note that there should be model files (`*.pt`) in `./model`.

## Cite the paper



## Reference

- https://github.com/zcaicaros/L2D
- https://github.com/yd-kwon/MatNet
- https://github.com/dmlc/dgl/tree/master/examples/pytorch/han