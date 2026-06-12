import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional  as F_trans
from torch.autograd import Variable
# from clip import clip
from math import exp
import numpy as np
from PIL import Image

class SobelxyRGB(nn.Module):
    def __init__(self,isSignGrad=True):
        super(SobelxyRGB, self).__init__()
        self.isSignGrad = isSignGrad
        kernelx = [[-0.2, 0, 0.2],
                  [-1, 0 , 1],
                  [-0.2, 0, 0.2]]
        kernely = [[0.2, 1, 0.2],
                  [0, 0 , 0],
                  [-0.2, -1, -0.2]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        kernelx = kernelx*1
        kernely = kernely*1
        kernelx = kernelx.repeat(1,3,1,1)
        kernely = kernely.repeat(1,3,1,1)
        # self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        # self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.register_buffer("weightx", kernelx)
        self.register_buffer("weighty", kernely)
        self.relu = nn.ReLU()

    def forward(self,x):
        #R,G,B = x[:,0,:,:],x[:,1,:,:],x[:,2,:,:]
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        if self.isSignGrad:
            return sobelx+sobely
        else:
            return torch.abs(sobelx)+torch.abs(sobely)



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        # self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        # self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.register_buffer("weightx", kernelx)
        self.register_buffer("weighty", kernely)

    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class MaxGradLoss(nn.Module):
    """Loss function for the grad loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0,isSignGrad=True):
        super(MaxGradLoss, self).__init__()
        self.loss_weight = loss_weight
        self.sobelconv = Sobelxy()
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir, *args, **kwargs):
        """Forward function.

        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): TIR image with shape (N, C, H, W).
        """        
        if im_tir!=None:
            rgb_grad = self.sobelconv(im_rgb)
            tir_grad = self.sobelconv(im_tir)

            mask = torch.ge(torch.abs(rgb_grad),torch.abs(tir_grad))
            max_grad_joint = tir_grad.masked_fill_(mask, 0) + rgb_grad.masked_fill_(~mask, 0)
            
            generate_img_grad = self.sobelconv(im_fusion)

            sobel_loss = self.L1_loss(generate_img_grad, max_grad_joint)
            loss_grad = self.loss_weight * sobel_loss
        else:
            rgb_grad = self.sobelconv(im_rgb)
            generate_img_grad = self.sobelconv(im_fusion)
            sobel_loss = self.L1_loss(generate_img_grad,rgb_grad)
            loss_grad = self.loss_weight * sobel_loss

        return loss_grad



def to_gray(img):
        #print(img.shape)
        r, g, b = img.unbind(dim=-3)
        # This implementation closely follows the TF one:
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        #print("l_imgshape",l_img.shape)
        l_img = l_img.unsqueeze(dim=-3)
        #print("l_imgshape",l_img.shape)
        return l_img


class MaxPixelLoss(nn.Module):
    """Loss function for the pixcel loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0):
        super(MaxPixelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir):
        """Forward function.
        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): RGB image with shape (N, C, H, W).
        """
        #print("im_tir",im_tir)
        if im_tir!=None:
            pixel_max = torch.max(im_rgb, im_tir).detach()
            #pixel_mean = (im_rgb + im_tir)/2.0
            pixel_loss = self.loss_weight*self.L1_loss(im_fusion,pixel_max)
        else:
            pixel_loss = self.loss_weight*self.L1_loss(im_fusion,im_rgb)
        
        return pixel_loss

    def getmaxpixel(self, im_rgb, im_tir,im_fusion):
        pixel_max = torch.max(im_rgb, im_tir)
        return  im_rgb, im_tir,pixel_max

class PixelLoss(nn.Module):
    """Loss function for the pixcel loss.

    Args:
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self, loss_weight=1.0):
        super(PixelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir):
        """Forward function.
        Args:
            im_fusion (Tensor): Fusion image with shape (N, C, H, W).
            im_rgb (Tensor): RGB image with shape (N, C, H, W).
        """
        #print("im_tir",im_tir)
        if im_tir!=None:
            #pixel_max = torch.max(im_rgb, im_tir).detach()
            pixel_mean = (im_rgb + im_tir)/2.0
            pixel_loss = self.loss_weight*self.L1_loss(im_fusion,pixel_mean)
        else:
            pixel_loss = self.loss_weight*self.L1_loss(im_fusion,im_rgb)
        
        return pixel_loss

class FusionStageLoss(nn.Module):

    def __init__(self, weigth = 1.0):
        super(FusionStageLoss, self).__init__()
        self.grad_loss = MaxGradLoss()
        self.pixel_loss = MaxPixelLoss()
        self.w = weigth

    def forward(self, im_fusion, im_rgb, im_tir):
        im_rgb = im_rgb.mean(dim=1, keepdim=True)
        im_tir = im_tir.mean(dim=1, keepdim=True)
        self.grad_loss = self.grad_loss.to(im_fusion.device)
        self.pixel_loss = self.pixel_loss.to(im_fusion.device)
        return self.grad_loss(im_fusion, im_rgb, im_tir) + self.w * self.pixel_loss(im_fusion, im_rgb, im_tir)