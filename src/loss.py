from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from options import args_parser

class PAAC(nn.Module, ABC):
    def __init__(self):
        super(PAAC, self).__init__()

    def forward(self, contrast_logits, contrast_target):
        loss_pca = F.cross_entropy(contrast_logits, contrast_target.long())
        return loss_pca


class PCD(nn.Module, ABC):
    def __init__(self):
        super(PCD, self).__init__()

    def forward(self, contrast_logits, contrast_target):
        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_pcd = (1 - logits).pow(2).mean()

        return loss_pcd

class PixelPrototypeCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELoss, self).__init__()
        self.seg_criterion = nn.CrossEntropyLoss()
        self.pca_criterion = PAAC()
        self.pcd_criterion = PCD()
        self.args = args_parser()
        self.loss_pca_weight = self.args.lambda1
        self.loss_pcd_weight = self.args.lambda2

    def forward(self, preds, target):
        if target.dim() == 4:
            target = target.squeeze(1)
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_pca = self.pca_criterion(contrast_logits, contrast_target)
            loss_pcd = self.pcd_criterion(contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)

            loss = self.seg_criterion(pred, target.long())
            return self.args.lambda0 * loss + self.loss_pca_weight * loss_pca + self.loss_pcd_weight * loss_pcd

        seg = preds
        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target.long())
        return loss