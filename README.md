# 3D Graph Neural Network with Few-Shot Learning for Predicting Drug-Drug Interactions in Scaffold-based Cold Start Scenarios

## Introduction

Welcome to the code repository of Meta3D-DDI. This repository includes code and datasets for training Meta3D-DDI models. Meta3D-DDI is a 3D graph neural network with few-shot learning, which is used to predict DDI events in cold start scenarios. In addition, the existing drug-based cold start setting may cause the scaffold structure information in the training set to leak into the test set. We design scaffold-based cold start scenarios to ensure that the drug scaffolds in the training set and test set don’t overlap.


## Datasets

We provide the DrugBank dataset in the datasets folder directly in this repo. However, due to Twosides being substantially larger than github's limit, we chose to upload it using zip compression. 
## Overview of code:

- datasets folder: Contains the dataset zip files and folders.
- config: Contains configuration files for each and every experiment listed in the experiments script folder.
- utils: Contains utilities for dataset extraction, parser argument extraction and storage of statistics and others.
- data.py: Contains the data providers for the few shot meta learning task generation. The data provider is agnostic to dataset, which means it can be used with any dataset. Most importantly, it can only scan and use datasets when they are presented in a specific format.
- experiment_builder.py: Builds an experiment ready to train and evaluate your meta learning models. It supports automatic
checkpoining and even fault-tolerant code. If your script is killed for whatever reason, you can simply rerun the script.
It will find where it was before it was killed and continue onwards towards convergence!

- few_shot_learning_system.py: Contains the meta_learning_system class which is where most of MAML and MAML++ are actually
implemented. It takes care of inner and outer loop optimization, checkpointing, reloading and statistics generation, as 
well as setting the rng seeds in pytorch.

- models: Contains new pytorch layers which are capable of utilizing either internal 
parameter or externally passed parameters. This is very useful in a meta-learning setting where inner-loop update 
steps are applied on the internal parameters. By allowing layers to receive weight which they will only use for the 
current inference phase, one can easily build various meta-learning models, which require inner_loop optimization 
without having to reload the internal parameters at every step. Essentially at the technical level, the meta-layers 
forward prop looks like:

```python
def forward(x, weights=None):
    if weights is not None:
        out = layer(x, weights)
    else:
        out = layer(x, self.parameters)
    return out
```
If we pass weights to it, then the layer/model will use those to do inference, otherwise it will use its internal 
parameters. Doing so allows a model like MAML to be build very easily. At the first step, use weights=None and for any
subsequent step just pass the new inner loop/dynamic weights to the network.

- train_maml_system.py: A very minimal script that combines the data provider with a meta learning system and sends them
 to the experiment builder to run an experiment. Also takes care of automated extraction of data if they are not 
 available in a folder structure.

# Running an experiment

## 1. Data preprocessing
The datasets folder contains CSV files that have been divided. If you want to repeat this work, you need to preprocess the data.

For DrugBank dataset
```python
python drugbank_split.ipynb
```
For TWOSIDES dataset
```python
python twoside_split.ipynb
```
## 2. Run
The code is based off the public code of [MAML++](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch), where their reimplementation of MAML is used as the baseline.

To run an experiment from the paper on DrugBank:
```python
python train_maml_system.py --name_of_args_json_file  config/drugbank_maml++.json 
```

To run an experiment from the paper on TWOSIDES:

```python
python train_maml_system.py --name_of_args_json_file  config/twoside_maml++.json 
```

# Reference
Qiujie Lv, Jun Zhou, Ziduo Yang, Haohuai He, and Calvin Yu-Chian Chen, 3D Graph Neural Network with Few-Shot Learning for Predicting Drug-Drug Interactions in Scaffold-based Cold Start Scenarios
 
