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


class DSQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_bit = 4, QInput = True, bSetQ = True):
        super(DSQLinear, self).__init__(in_features, out_features, bias=bias)
        self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.num_bit -1	 
        self.is_quan = bSetQ
        if self.is_quan:
            # using int32 max/min as init and backprogation to optimization
            # Weight
            self.uW = nn.Parameter(data = torch.tensor(2 **31 - 1).float())
            self.lW  = nn.Parameter(data = torch.tensor((-1) * (2**32)).float())
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
        alpha = alpha.clamp(None, 2)
        s = 1/(1-alpha)
        k = (1/delta) * (2/alpha - 1).log()

        out = s * ((k * (x - mi)).tanh())

        return out  

    def sgn(self, x):
        # x = torch.where(x>=0, 1.0, -1.0)
        # where does support autograd
        # use normolize and round instead
        delta = torch.max(x) - torch.min(x)
        x = ((x - torch.min(x))/delta)
        x = RoundWithGradient.apply(x) * 2 -1

        return x

    def dequantize(self, x, lower_bound, delta, interval):

        out = lower_bound + delta * (interval + (x+1)/2)

        return out

    def forward(self, x):
        if self.is_quan:
            # Weight Part
            Qweight = torch.where(self.weight >= self.uW, self.uW, self.weight)
            Qweight = torch.where(Qweight <= self.lW, self.lW, Qweight)
            cur_max = torch.max(Qweight)
            cur_min = torch.min(Qweight)
            delta =  (cur_max - cur_min)/(self.bit_range)
            interval = (Qweight - cur_min) //delta            
            mi = (interval + 0.5) * delta + cur_min
            Qweight = self.phi_function(Qweight, mi, self.alphaW, delta)
            Qweight = self.sgn(Qweight)
            Qweight = self.dequantize(Qweight, cur_min, delta, interval)

            Qbias = None
            # Bias			
            if self.bias is not None:
                Qbias = torch.where(self.bias >= self.ub, self.ub, self.bias)
                Qbias = torch.where(Qbias <= self.lb, self.lb, Qbias)
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
                Qactivation = torch.where(x >= self.uA, self.uA, x)
                Qactivation = torch.where(Qactivation <= self.lA, self.lA, Qactivation)
                cur_max = torch.max(Qactivation)
                cur_min = torch.min(Qactivation)
                delta =  (cur_max - cur_min)/(self.bit_range)
                interval = (Qactivation - cur_min) //delta
                mi = (interval + 0.5) * delta + cur_min
                Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                Qactivation = self.sgn(Qactivation)
                Qactivation = self.dequantize(Qactivation, cur_min, delta, interval)
                                            
            output = F.linear(Qactivation, Qweight, Qbias)
            
        else:
            output =  F.linear(x, self.weight, self.bias)

        return output