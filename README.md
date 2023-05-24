If you use or adopt any of the work presented in this repository, please cite our work as follows:

```
@article{Stewart_2022,
doi = {10.1088/2634-4386/ac8828},
url = {https://dx.doi.org/10.1088/2634-4386/ac8828},
year = {2022},
month = {sep},
publisher = {IOP Publishing},
volume = {2},
number = {4},
pages = {044002},
author = {Kenneth M Stewart and Emre O Neftci},
title = {Meta-learning spiking neural networks with surrogate gradient descent},
journal = {Neuromorphic Computing and Engineering }
}
```

# Introduction

This repository has been modified from pytorch-maml (https://github.com/tristandeleu/pytorch-maml)

# Usage

## Notes for setup and data acquisition

The datasets for Double NMNIST, Double ASL-DVS, and N-Omniglot as detailed in the paper are obtainable and created with the torchneuromorphic repository linked here:
https://github.com/kennetms/torchneuromorphic


## Basic Usage for MAML ANN
```
run train_wandb.py --benchmark=omniglot --no-log 
```
The option no-log here disables wand.ai logging.

## Basic Usage for MAML SNN

For more details on the background for MAML SNN, see pre-print https://arxiv.org/abs/2201.10777

### Example run of double nmnist (if not loading model omit --load-model):
```
python train.py --output-folder='logs/doublenmnistsequence' --benchmark='doublenmnistsequence' --batch-size=1 --verbose --meta-lr=.002 --step-size=1 --num-steps=1 --num-workers=10 --params_file='parameters/decolle_params-CNN.yml' --num-shots=1 --load-model=double_nmnist_sequence_best/best_model/model.th --num-batches=200 --num-batches-test=20 --num-epochs=100 
```

### Example run of detaching the last layer (add --detach-at=):
```
python train.py --output-folder='logs/doublenmnistsequence' --benchmark='doublenmnistsequence' --batch-size=1 --verbose --meta-lr=.002 --step-size=1 --num-steps=1 --num-workers=10 --params_file='parameters/decolle_params-CNN.yml' --num-shots=1 --load-model=double_nmnist_sequence_best/best_model/model.th --num-batches=200 --num-batches-test=20 --num-epochs=10 --device=1 --do-test --detach-at=0
```

### Example run of double asl dvs (if not loading model omit --load-model):
```
python train.py --output-folder='logs/doubledvssignsequence' --benchmark='doubledvssignsequence' --batch-size=1 --verbose --meta-lr=.002 --step-size=1 --num-steps=1 --num-workers=10 --params_file='parameters/decolle_params-CNN-Sign.yml' --num-shots=1 --num-batches=200 --num-batches-test=20 --num-epochs=100 --load-model=logs/doubledvssignsequence/2021-12-10_201651/model.th
```

 
```
## Licensing
These assets are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt).

