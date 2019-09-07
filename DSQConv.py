import torch
import torch.nn as nn
import torch.nn.functional as F

class DSQConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_bit = 8, QInput = False):
        super(DSQConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		self.num_bit = num_bit
        self.quan_input = QInput
        self.bit_range = 2**self.bit_num -1	 
        # using int32 max/min as init and backprogation to optimization
        # Weight
		self.uW = nn.Parameter(data = torch.tensor(2 **31 - 1))
		self.lW  = nn.Parameter(data = torch.tensor((-1) * (2**32)))
        self.alphaW = nn.Parameter(data = torch.tensor(0.2))
        # Bias
		if self.bias is not None
			self.uB = nn.Parameter(data = torch.tensor(2 **31 - 1))
			self.lB  = nn.Parameter(data = torch.tensor((-1) * (2**32)))
			self.alphaB = nn.Parameter(data = torch.tensor(0.2))
   
        # Activation input		
        if self.quan_input:
		self.uA = nn.Parameter(data = torch.tensor(2 **31 - 1))
		self.lA  = nn.Parameter(data = torch.tensor((-1) * (2**32)))
		self.alphaA = nn.Parameter(data = torch.tensor(0.2))
  
    def phi_function(x, mi, alpha, delta):			
		# alpha should less than 2 or log will be None
		alpha = alpha.clamp(None, 2)
		s = 1/(1-alpha)
		k = (1/delta) * (2/alpha - 1).log()
		
		out = s * ((k * (x - mi)).tanh())

		return out	
	
	def sgn(x):
		# x = torch.where(x>=0, 1.0, -1.0)
        # where does support autograd
        # use normolize and round instead
        delta = torch.max(x) - torch.min(x)
        x = ((x - torch.min(x))/delta).round() * 2 - 1
		
		return x
	
	def dequantize(x, lower_bound, delta, interval):
		
		out = lower_bound + delta * (interval + (x+1)/2)
		
		return out

    def forward(x):

        # Weight Part        
        Qweight = self.weight.clamp(self.lW, self.uW)    
		deltaW = (self.uW - self.lW)/(self.bit_range)
        W_interval = Qweight//deltaW
        W_mi = self.uW + (W_interval + 0.5) * deltaW
        Qweight = self.phi_function(Qweight, W_mi, self.alphaW, deltaW)
        Qweight = self.sgn(Qweight)
        Qweight = self.dequantize(Qweight, self.lW, deltaW, W_interval)
        
        # Bias			
        if self.bias is not None:
            Qbias = self.weight.clamp(self.lB, self.uB)    
            deltaB = (self.uB - self.lB)/(self.bit_range)
            B_interval = Qbias//deltaB
            B_mi = self.uB + (B_interval + 0.5) * deltaB
            Qbias = self.phi_function(Qbias, B_mi, self.alphaB, deltaB)
            Qbias = self.sgn(Qbias)
            Qbias = self.dequantize(Qbias, self.lB, deltaB, B_interval)
        
        # Input(Activation)
        Qactivation = x
        if self.quan_input:
            Qactivation = x.clamp(self.lA, self.uA)    
            deltaA = (self.uA - self.lA)/(self.bit_range)
            A_interval = Qactivation//deltaA
            A_mi = self.uA + (A_interval + 0.5) * deltaA
            Qactivation = self.phi_function(Qactivation, A_mi, self.alphaA, deltaA)
            Qactivation = self.sgn(Qactivation)
            Qactivation = self.dequantize(Qactivation, self.lA, deltaA, A_interval)
                
        output = F.conv2d(Qactivation, Qweight, Qbias,  self.stride, self.padding, self.dilation, self.groups)
        
        return output