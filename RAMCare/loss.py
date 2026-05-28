import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class RESTORE_LOSS(nn.Module):
    def __init__(self):
        super(RESTORE_LOSS, self).__init__()

    def forward(self, recon, label, mask=None):
        if mask == None:
            mse_per_sample = ((recon - label) ** 2).mean(dim=1)
        else:
            pass
            mse_per_sample = torch.sum(((recon - label) ** 2) * mask, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))

        return mse_per_sample  # B