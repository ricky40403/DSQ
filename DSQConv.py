import torch
import torch.nn as nn
import torch.nn.functional as F


class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g 


class DSQConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_bit = 8, QInput = True, bSetQ = True):
        super(DSQConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1	 
        self.is_quan = bSetQ
        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.uW = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
            self.lW  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
            self.register_buffer('running_uw')
            self.register_buffer('running_uw')
            self.alphaW = nn.Parameter(data = torch.tensor(0.2).float())
            # Bias
            if self.bias is not None:
                self.uB = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.lB  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
                self.alphaB = nn.Parameter(data = torch.tensor(0.2).float())

            # Activation input		
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.lA  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
                self.alphaA = nn.Parameter(data = torch.tensor(0.2).float())



    def phi_function(self, x, mi, alpha, delta):

        # alpha should less than 2 or log will be None
        # alpha = alpha.clamp(None, 2)
        alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
        s = 1/(1-alpha)
        k = (2/alpha - 1).log() * (1/delta)
        x = (((x - mi) *k ).tanh()) * s 
        return x	

    def sgn(self, x):
        # x = torch.where(x>=0, 1.0, -1.0)
        # where does support autograd
        # use normolize and round instead
        delta = torch.max(x) - torch.min(x)
        x = (x/delta + 0.5)
        # x = ((x - torch.min(x))/delta)
        # x.sub_(torch.min(x)).div_(delta)
        x = RoundWithGradient.apply(x) * 2 -1

        return x

    def dequantize(self, x, lower_bound, delta, interval):

        # save mem
        x =  ((x+1)/2 + interval) * delta + lower_bound
        # x.add_(1).div_(2).add_(interval).mul_(delta).add_(lower_bound)

        return x

    def forward(self, x):
        if self.is_quan:
            # Weight Part
            Qweight = torch.where(self.weight >= self.uW, self.uW, self.weight)
            Qweight = torch.where(Qweight <= self.lW, self.lW, Qweight)
            Qweight = self.weight
            cur_max = torch.max(Qweight)
            cur_min = torch.min(Qweight)
            delta =  (cur_max - cur_min)/(self.bit_range)
            interval = (Qweight - cur_min) //delta            
            mi = (interval + 0.5) * delta + cur_min
            Qweight = self.phi_function(Qweight, mi, self.alphaW, delta)
            Qweight = self.sgn(Qweight)
            Qweight = self.dequantize(Qweight, cur_min, delta, interval)

            Qbias = self.bias
            # Bias			
            if self.bias is not None:
                Qbias = torch.where(self.bias >= self.ub, self.ub, self.bias)
                Qbias = torch.where(Qbias <= self.lb, self.lb, Qbias)
                Qbias = self.bias
                cur_max = torch.max(Qbias)
                cur_min = torch.min(Qbias)
                delta =  (cur_max - cur_min)/(self.bit_range)
                interval = (Qbias - cur_min) //delta
                mi = (interval + 0.5) * delta + cur_min
                Qbias = self.phi_function(Qbias, mi, self.alphaB, delta)
                Qbias = self.sgn(Qbias)
                Qbias = self.dequantize(Qbias, cur_min, delta, interval)

            # # Input(Activation)
            Qactivation = x
            # if self.quan_input:
            #     # print("QQQQQ INput")       
            #     Qactivation = torch.where(x >= self.uA, self.uA, x)
            #     Qactivation = torch.where(Qactivation <= self.lA, self.lA, Qactivation)                
            #     cur_max = torch.max(Qactivation)
            #     cur_min = torch.min(Qactivation)
            #     delta =  (cur_max - cur_min)/(self.bit_range)
            #     interval = (Qactivation - cur_min) //delta
            #     mi = (interval + 0.5) * delta + cur_min
            #     Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
            #     Qactivation = self.sgn(Qactivation)
            #     Qactivation = self.dequantize(Qactivation, cur_min, delta, interval)
           
            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output