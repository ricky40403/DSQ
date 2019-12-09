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
                momentum = 0.1,                
                num_bit = 8, QInput = True, bSetQ = True):
        super(DSQConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1	 
        self.is_quan = bSetQ        
        self.momentum = momentum
        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.uW = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
            self.lW  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
            self.register_buffer('running_uw', self.uw.data) # init with uw
            self.register_buffer('running_lw', self.lw.data) # init with lw
            self.alphaW = nn.Parameter(data = torch.tensor(0.2).float())
            # Bias
            if self.bias is not None:
                self.uB = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.lB  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
                self.register_buffer('running_uB', self.ub.data)# init with ub
                self.register_buffer('running_lB', self.lb.data)# init with lb
                self.alphaB = nn.Parameter(data = torch.tensor(0.2).float())
                

            # Activation input		
            if self.quan_input:
                self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
                self.lA  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
                self.register_buffer('running_uA', self.uA.data) # init with uA
                self.register_buffer('running_lA', self.lA.data) # init with lA
                self.alphaA = nn.Parameter(data = torch.tensor(0.2).float())

    def clipping(self, x, upper, lower):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x

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
            # moving average
            self.running_lw.mul_(1-self.momentum).add_((self.momentum) * self.lW)
            self.running_uw.mul_(1-self.momentum).add_((self.momentum) * self.uW)
            Qweight = self.clipping(self.weight, self.self.running_uw, self.self.running_lw)
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
                self.running_lB.mul_(1-self.momentum).add_((self.momentum) * self.lB)
                self.running_uB.mul_(1-self.momentum).add_((self.momentum) * self.uB)
                Qbias = self.clipping(self.bias, self.running_uB, self.running_lB)
                cur_max = torch.max(Qbias)
                cur_min = torch.min(Qbias)
                delta =  (cur_max - cur_min)/(self.bit_range)
                interval = (Qbias - cur_min) //delta
                mi = (interval + 0.5) * delta + cur_min
                Qbias = self.phi_function(Qbias, mi, self.alphaB, delta)
                Qbias = self.sgn(Qbias)
                Qbias = self.dequantize(Qbias, cur_min, delta, interval)

            # Input(Activation)
            Qactivation = x
            if self.quan_input:
                self.running_lA.mul_(1-self.momentum).add_((self.momentum) * self.lA)
                self.running_uA.mul_(1-self.momentum).add_((self.momentum) * self.uA)
                Qactivation = self.clipping(x, self.running_uA, self.running_lA)
                cur_max = torch.max(Qactivation)
                cur_min = torch.min(Qactivation)
                delta =  (cur_max - cur_min)/(self.bit_range)
                interval = (Qactivation - cur_min) //delta
                mi = (interval + 0.5) * delta + cur_min                
                Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                Qactivation = self.sgn(Qactivation)
                Qactivation = self.dequantize(Qactivation, cur_min, delta, interval)
            
            
            # sys.exit()
            output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)

        else:
            output =  F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return output