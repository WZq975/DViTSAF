# Distilling Vision Transformer via Self-Attention Forcing (DViTSAF)


## Usage

To run the project, use the following command:
```
train.py [-h] [--objective {teacher,student,distillation}] [--distillation {KD,SAF,SAF+KD}] [--dataset {CIFAR-10,CIFAR-100,Flowers-102}] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--logdir LOGDIR]
```
The options for running the script are as follows:

### Model

* `--objective {teacher,student,distillation}`: The type of model to train, whether teacher, student or distillation (default: distillation).
* `--distillation {KD,SAF,SAF+KD}`: The type of distillation method to use: KD, SAF(ours), SAF+KD (default: SAF).

### Dataset

* `--dataset {CIFAR-10,CIFAR-100,Flowers-102}`: The name of the dataset to use (default: CIFAR10).

### Data

* `--batch_size BATCH_SIZE`: The batch size to use for training (default: 256).

### Optimization

* `--epochs EPOCHS`: The number of epochs to train for (default: 40).
* `--lr LR`: The learning rate to use for the Adam optimizer (default: 0.0001).

### Experiment Config

* `--logdir LOGDIR`: A unique experiment identifier to use for logging (default: ./logs).

## Options

* `-h, --help`: Shows the help message and exits.

## Examples

Here are examples of how to run the scripts to reproduct the results on CIFAR-100:

Train the teacher:
```
python train.py --objective 'teacher' --dataset 'CIFAR-100' --batch_size 64
```
Distillation via SAF:
```
python train.py --objective 'distillation' --distillation 'SAF' --dataset 'CIFAR-100' --batch_size 256
```
Directly train the student without distillation:
```
python train.py --objective 'student' --dataset 'CIFAR-100' --batch_size 64
```
