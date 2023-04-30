# Distilling Vision Transformer via Self-Attention Forcing (DViTSAF)
| Model            | CIFAR-10 | CIFAR-100 | Flowers-102 |
| ----------------|---------:|----------:|------------:|
| Teacher (ViT-L)  | 98.23%   | 89.29%    | 98.70%      |
| Student (ViT-T)  | 96.25%   | 84.67%    | 84.32%      |
| DViTSAF (Ours)   | [**97.10%**](https://drive.google.com/file/d/1BUCs2YZykusyXLMYwaniyQ9wuOBKyKpN/view?usp=share_link)   | [**85.74%**](https://drive.google.com/file/d/18BM9S7D_MD0GLamvrwVqXhaqjZRteJTy/view?usp=share_link)    | [**88.45%**](https://drive.google.com/file/d/1jOM8HH2vCdPAAIw9o6F4DDanPluLOuy0/view?usp=share_link)      |
| Vanilla KD       | 96.33%   | 85.02%    | 87.67%      |
| DViTSAF + Vanilla KD | [**97.43%**](https://drive.google.com/file/d/1D1KIv1Q0u5oQlKgzkO3bIv5pSEBRTtSo/view?usp=sharing) | [**85.97%**](https://drive.google.com/file/d/1AvKRNkxzUlawEwDFlFgVJpWFk1m2DFYW/view?usp=share_link) | [**89.09%**](https://drive.google.com/file/d/1aJ6oh-daYk7N2Ztg2xAgcRq4lJAU2zyK/view?usp=share_link) |

The trained weights have been hyperlinked to their corresponding accuracies.

## Usage

To run the project, use the following command:
```
python train.py [-h] [--objective {teacher,student,distillation}] [--distillation {KD,SAF,SAF+KD}] [--dataset {CIFAR-10,CIFAR-100,Flowers-102}] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--logdir LOGDIR]
```
The options for running the script are as follows:

#### Model

* `--objective {teacher,student,distillation}`: The type of model to train, whether teacher, student or distillation (default: distillation).
* `--distillation {KD,SAF,SAF+KD}`: The type of distillation method to use: KD, SAF(ours), SAF+KD (default: SAF).

#### Dataset

* `--dataset {CIFAR-10,CIFAR-100,Flowers-102}`: The name of the dataset to use (default: CIFAR10).

#### Data

* `--batch_size BATCH_SIZE`: The batch size to use for training (default: 256).

#### Optimization

* `--epochs EPOCHS`: The number of epochs to train for (default: 40).
* `--lr LR`: The learning rate to use for the Adam optimizer (default: 0.0001).

#### Experiment Config

* `--logdir LOGDIR`: A unique experiment identifier to use for logging (default: ./logs).


## Examples

Here are examples of how to run the scripts to reproduce the above results on CIFAR-100:

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
