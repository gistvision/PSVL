import numpy as np
import torch
from torch import nn
import torch

class TAGLoss(nn.Module):
    def __init__(self):
        super(TAGLoss, self).__init__()

    def forward(self, w, mask):
        ac_loss = (-mask*torch.log(w+1e-8)).sum(1) / mask.sum(1)
        ac_loss = ac_loss.mean(0)

        return ac_loss

class TGRegressionCriterion(nn.Module):
    def __init__(self):
        super(TGRegressionCriterion, self).__init__()

        self.regloss1 = nn.SmoothL1Loss()
        self.regloss2 = nn.SmoothL1Loss()

    def forward(self, loc, s_gt, e_gt):

        total_loss = self.regloss1(loc[:,0], s_gt) + self.regloss2(loc[:,1], e_gt)

        return total_loss

class NLVLLoss(nn.Module):
    def __init__(self,cfg, reg_w=1):
        super().__init__()
        self.temporal_localization_loss = TGRegressionCriterion()
        self.temporal_attention_loss2 = TAGLoss()
        self.reg_w = reg_w

    def forward(self,model_outputs,batch):
        # position loss
        timestamps = model_outputs['timestamps'] # [B,2]
        gt_start_pos = batch["grounding_start_pos"]
        gt_end_pos = batch["grounding_end_pos"]
        gt_timestamps = torch.cat([gt_start_pos.unsqueeze(1),gt_end_pos.unsqueeze(1)],dim=1) # [B,2]

        localization_loss = self.temporal_localization_loss(timestamps, gt_start_pos, gt_end_pos)
        localization_loss = localization_loss * self.reg_w

        # attention loss
        attention_weights = model_outputs['attention_weights']  # [B,128]
        attention_masks = batch["attention_masks"]              # [B,128] 
        attention_loss = self.temporal_attention_loss2(attention_weights,attention_masks)
        
        loss_dict = {
            "localization_loss": localization_loss,
            "attention_loss": attention_loss
        }
        return loss_dict
