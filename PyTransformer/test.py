import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import copy
from transformers.torchTransformer import TorchTransformer
from transformers.quantize import QConv2d
model = models.__dict__["resnet18"]()
model.cuda()
model = model.eval()

transofrmer = TorchTransformer()
transofrmer.register(nn.Conv2d, QConv2d)
model = transofrmer.trans_layers(model)
print(model)
sys.exit()


input_tensor = torch.randn([1, 3, 224, 224])		
input_tensor = input_tensor.cuda()
net = transofrmer.summary(model, input_tensor=input_tensor)
# transofrmer.visualize(model, input_tensor = input_tensor, save_name= "example", graph_size = 80)