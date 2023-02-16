import torch.nn as nn
import torch.utils.data
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from .base_model import BaseModel
from utils.utils import AverageMeter
import os


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net_net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net_net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        out = self.active(out)

        return out

class U_Net(nn.Module, BaseModel):
    def __init__(self, args) -> None:
        super(U_Net, self).__init__()
        BaseModel.__init__(self, args)
        
        self.net = U_Net_net(3,3)
        self.nets.append(self.net)
        
        self.optimizer =optim.SGD(
            self.net.parameters(), 
            lr=args.lr,
            momentum=args.momentum, 
            nesterov=True, 
            weight_decay=args.weight_decay)

        self.scheduler = MultiStepLR(self.optimizer, args.milestones, args.gamma)
        self.schedulers.append(self.scheduler)
        self.loss_function_mse = nn.MSELoss(reduction='mean')
        
        self.loss_names += ['mse']
        
        self.meter_init()

        
    def to_cuda(self):
        self.net = self.net.cuda()
        self.loss_function_mse = self.loss_function_mse.cuda()
        return self
        
    def set_input(self, data):
        self.blur_images, self.sharp_images = data
        self.blur_images = self.blur_images.cuda()
        self.sharp_images = self.sharp_images.cuda()
        
    def forward(self):
        self.restored_images = self.net(self.blur_images)
        
    
        
    def train_step(self):        
        self.forward()
        self.train_loss_mse = self.loss_function_mse(self.restored_images, self.sharp_images)
        
        self.train_loss_all = self.train_loss_mse
        
        self.optimizer.zero_grad()
        self.train_loss_all.backward()
        self.optimizer.step()

        self.update_meters(True, self.blur_images.size(0))
        
    def valid_step(self):
        with torch.no_grad():
            self.forward()
            self.valid_loss_mse = self.loss_function_mse(self.restored_images, self.sharp_images)
            
            self.valid_loss_all = self.valid_loss_mse
            
            self.update_meters(False, self.blur_images.size(0))

    def load_network(self):
        ckpt = torch.load(self.args.load_ckpt_path)
        self.net.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])

if __name__ == '__main__':
    model = U_Net(3,3)
    
    input = torch.randn((4,3,224,224))
    
    output = model(input)
    
    print(output.shape)