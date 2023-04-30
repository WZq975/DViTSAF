# Distilling Vision Transformer via Self-Attention Forcing

This project provides an implementation of distillation of Vision Transformers, allowing for the training of smaller student models to match the performance of larger teacher models. 

## Usage

To run the project, use the following command:

train.py [-h] [--batch_size BATCH_SIZE] [--objective {teacher,student,distillation}] [--distillation {KD,SAF,SAF+KD}] [--epochs EPOCHS] [--lr LR] [--dataset {CIFAR-10,CIFAR-100,Flowers-102}] [--logdir LOGDIR]

The options for running the script are as follows:

### Data

* `--batch_size BATCH_SIZE`: The batch size to use for training (default: 256).

### Model

* `--objective {teacher,student,distillation}`: The type of model to train, whether teacher, student or distillation (default: distillation).
* `--distillation {KD,SAF,SAF+KD}`: The type of distillation method to use: KD, SAF(ours), SAF+KD (default: SAF).

### Optimization

* `--epochs EPOCHS`: The number of epochs to train for (default: 40).
* `--lr LR`: The learning rate to use for the Adam optimizer (default: 0.0001).

### Dataset

* `--dataset {CIFAR-10,CIFAR-100,Flowers-102}`: The name of the dataset to use (default: CIFAR10).

### Experiment Config

* `--logdir LOGDIR`: A unique experiment identifier to use for logging (default: ./logs).

## Options

* `-h, --help`: Shows the help message and exits.

## Example

Here is an example of how to run the script with some custom options:

train.py --objective student --distillation SAF --epochs 50 --lr 0.0005 --dataset CIFAR-100 --logdir ./mylogs


This will train a student model using the SAF distillation method for 50 epochs, with a learning rate of 0.0005, on the CIFAR-100 dataset. The logs will be saved to a folder called "mylogs".
