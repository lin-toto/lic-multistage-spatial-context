# Multistage Spatial Context Models for Learned Image Compression

This is the official code repository for the paper 
[Multistage Spatial Context Models for Learned Image Compression](https://arxiv.org/abs/2302.09263).

This implementation is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI/). 
Please check the commit history for detailed change logs. 

This repository contains:
- CompressAI commit hash [b10cc7c](https://github.com/InterDigitalInc/CompressAI/commit/b10cc7c1c51a0af26ea5deae474acfd5afdc1454) 
- An efficient Gaussian Mixture Model implementation
- Multistage Spatial Context Model implementations on Cheng2020

Pretrained weights for MSE are available at 
[this repository](https://github.com/lin-toto/lic-multistage-spatial-context-pretrained).

## Usage

### Installation
Clone this repository and install with:
```bash
python3 setup.py install
```

You may have to remove your existing CompressAI installation first as this implementation replaces it. 
We recommend creating a new virtual environment for testing this implementation.

### Evaluation
Use the CompressAI scripts for evaluating the models:
```bash
python3 -m compressai.utils.eval_model pretrained /path/to/dataset -a cheng2020-attn-gmm-multistage-{2x2,4x4} -m mse -q {1-6}
```

