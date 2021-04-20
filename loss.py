import numpy as np

import torch
import torch.nn as nn
# from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence


class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


# class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
#     def __init__(self):
#         super().__init__()
#         self.name = "ChamferLoss"

#     def forward(self, bins, target_depth_maps):
#         bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
#         n, p = bin_centers.shape
#         input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
#         # n, c, h, w = target_depth_maps.shape

#         target_points = target_depth_maps.flatten(1)  # n, hwc
#         mask = target_points.ge(1e-3)  # only valid ground truth points
#         target_points = [p[m] for p, m in zip(target_points, mask)]
#         target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
#         target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

#         loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
#         return loss

# Modification below
# Refer to "Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries" (https://arxiv.org/abs/1803.08673)
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation
class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "DepthLoss"
    
    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        loss_depth = torch.log(torch.abs(input - target) + 0.5).mean() 

        return loss_depth

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        n, c, h, w = x.shape
            x is n examples, each have h*w pixels, and each pixel contain c=1 channel value
        
        n, 2, h, w = out.shape 
            2 channel: first represents dx, second represents dy
        """
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
        return out

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = ["GradientLoss", "NormLoss"]
        self.sobel = Sobel().cuda()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        ones = torch.ones(input.size(0), 1, input.size(2),input.size(3)).float().to(input.device)

        input_grad = self.sobel(input)
        target_grad = self.sobel(target)
        #  n, 2, h, w = out.shape 
        #  2 channel: first represents dx, second represents dy

        input_grad_dx = input_grad[:, 0, :, :].contiguous().view_as(input)
        input_grad_dy = input_grad[:, 1, :, :].contiguous().view_as(input)
        target_grad_dx = target_grad[:, 0, :, :].contiguous().view_as(target)
        target_grad_dy = target_grad[:, 1, :, :].contiguous().view_as(target)

        input_normal = torch.cat((-input_grad_dx, -input_grad_dy, ones), 1)
        target_normal = torch.cat((-target_grad_dx, -target_grad_dy, ones), 1)

        loss_dx = torch.log(torch.abs(input_grad_dx - target_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(input_grad_dy - target_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - self.cos(input_normal, target_normal)).mean()

        return loss_dx+loss_dy, loss_normal
