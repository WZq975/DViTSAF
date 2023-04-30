# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
(taken from https://github.com/facebookresearch/deit/blob/main/losses.py with modifications)
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                alpha: float, tau: float, distillation: str):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.tau = tau
        self.distillation = distillation
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, inputs, student_outputs, labels, student_weights):

        base_loss = self.base_criterion(student_outputs, labels)
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_outputs, teacher_weights = self.teacher_model(inputs)

        T = self.tau

        if self.distillation == "SAF" or self.distillation == "SAF+KD":
            # SAF
            distillation_loss_1 = []
            selected_teacher_weights = teacher_weights[len(teacher_weights)//len(student_weights)-1::len(teacher_weights)//len(student_weights)]
            # print(len(student_weights), len(teacher_weights), len(selected_teacher_weights))
            for i in range(len(student_weights)):
               distillation_loss_1.append(self.mse_loss(student_weights[i], selected_teacher_weights[i]))
            assert len(distillation_loss_1) == len(student_weights)
            # print(distillation_loss)
            distillation_loss_1 = torch.stack(distillation_loss_1)
            # print(distillation_loss)
            distillation_loss_1 = torch.sum(distillation_loss_1, dim=0)
            # print(distillation_loss)

        if self.distillation == "KD" or self.distillation == "SAF+KD":
            # KD
            distillation_loss_2 = F.kl_div(
                F.log_softmax(student_outputs / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / student_outputs.numel()

        if self.distillation == "KD":
            distillation_loss = distillation_loss_2
        elif self.distillation == "SAF":
            distillation_loss = distillation_loss_1
        elif self.distillation == "SAF+KD":
            distillation_loss = distillation_loss_1 + distillation_loss_2

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        # print(loss, base_loss, distillation_loss_1, distillation_loss_2)
        return loss, base_loss, distillation_loss
