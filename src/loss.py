"""Custom loss for long tail problem.

- Author: Junghoon Kim
- Email: placidus36@gmail.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCriterion:
    """Custom Criterion."""

    def __init__(self, samples_per_cls, device, fp16=False, loss_type="softmax"):
        if not samples_per_cls:
            loss_type = "softmax"
        else:
            self.samples_per_cls = samples_per_cls
            self.frequency_per_cls = samples_per_cls / np.sum(samples_per_cls)
            self.no_of_classes = len(samples_per_cls)
        self.device = device
        self.fp16 = fp16

        if loss_type == "softmax":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == "logit_adjustment_loss":
            tau = 1.0
            self.logit_adj_val = (
                torch.tensor(tau * np.log(self.frequency_per_cls))
                .float()
                .to(self.device)
            )
            self.logit_adj_val = (
                self.logit_adj_val.half() if fp16 else self.logit_adj_val.float()
            )
            self.logit_adj_val = self.logit_adj_val.to(device)
            self.criterion = self.logit_adjustment_loss

    def __call__(self, logits, labels):
        """Call criterion."""
        return self.criterion(logits, labels)

    def logit_adjustment_loss(self, logits, labels):
        """Logit adjustment loss."""
        logits_adjusted = logits + self.logit_adj_val.repeat(labels.shape[0], 1)
        loss = F.cross_entropy(input=logits_adjusted, target=labels)
        return loss


"""Knowledge Distillation
- Author: Sungjin Park, Sangwon Lee  
- Contact: 8639sung@gmail.com
"""
class CustomCriterion_KD:
    """Custom Criterion for Knowledge Distillation"""

    def __init__(self, samples_per_cls, device, fp16=False, loss_type="softmax"):
        self.criterion = self.knowledge_distillation_loss

    def __call__(self, logits, labels, teacher_logits=None):
        """Call criterion."""
        return self.criterion(logits, labels, teacher_logits) 

    def knowledge_distillation_loss(self, logits, labels, teacher_logits):
        """Logit adjustment loss."""
        alpha = 0.0
        T = 1
        loss = F.cross_entropy(input=logits, target=labels)

        if teacher_logits == None:
            return loss    
        else:
            D_KL = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (T * T)
            KD_loss =  (1. - alpha)*loss + alpha*D_KL

            return KD_loss