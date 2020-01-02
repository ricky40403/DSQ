# DSQ
pytorch unofficial implementation of "Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks"  


****
The Origin Paper : <https://arxiv.org/abs/1908.05033>  
****
This repository follow the Algorithm 1 in the paper.  

This repository uses the max value of int32 as the initial value.  
It should not affect the value range (because the parameter of the deep model should not too large), and most of the edge device range is up to int32.  

----

# Training
Training with quantization.
Scrip modified from <https://github.com/pytorch/examples/tree/master/imagenet>

Now support uniform/DSQ quantization  
adding argments  
-q : quantization type, default is None  
--quantize_input : quantize input or not  
--quan_bit : quantization bit num  
--log_path : tensorboard log path to write, default folder is ./log.  

Examples  
Training DSQ with 8 bit (no quantiza input)
```
python train.py -a resnet18  -q DSQ --quan_bit 8 {Path to data}
```

Training DSQ with 8 bit ( quantiza input)
```
python train.py -a resnet18  -q DSQ --quan_bit 8 --quantize_input {Path to data}
```

Evaluating (directly use evaluation and resume from model_best.pth.tar)
```
python train.py -a resnet18 -q DSQ --quan_bit 8 --quantize_input --resume {path to model_best.pth.tar} -- evaluate {Path to data}
```
# Experiments

The results is base on fake-quantization.(only quantized convolution).
As the mentioned in the paper, not to quantize the final Linear Layer.

<table>
<tr><th> model </th> <th> QuanType </th> <th> W/A bit </th> <th> top1 </th> <th> top5 </th></tr>  
<tr><th rowspan="4"> resnet18 </th> <th> UniformQuan </th> <th> 4/32 </th> <th> 69.486 </th> <th> 89.004 </th></tr>
<tr><th> DSQ </th> <th> 4/32 </th> <th> 69.328 </th> <th> 88.872 </th></tr>
<tr><th> UniformQuan </th> <th> 4/4 </th> <th> 69.306 </th> <th> 88.780 </th></tr>
<tr><th> DSQ </th> <th> 4/4 </th> <th> 69.542 </th><th> 88.884 </th></tr> 
</table>

learned alpha for 4 bit DSQ (quantize weight and input)  

layer           | weight  | activation|
----------------|:-------:|----------:|
layer1.0.conv1  |  0.4832 | 0.5661 |  
layer1.0.conv2  |  0.3730 | 0.2953 |  
layer1.1.conv1  |  0.4405 | 0.2975 |  
layer1.1.conv2  |  0.3427 | 0.1959 |  
layer2.0.conv1  |  0.3966 | 0.1653 |  
layer2.0.conv2  |  0.4140 | 0.2014 |  
layer2.downsample| **0.3275** | **0.1779** |  
layer2.1.conv1  |  0.4303 | 0.1675 |  
layer2.1.conv2  |  0.4207 | 0.1570 |  
layer3.0.conv1  |  0.4590 | 0.2774 |  
layer3.0.conv2  |  **0.4838** | **0.2569** |  
layer3.downsample|  **0.2305** | **0.1073** |  
layer3.1.conv1  |  0.4523 | 0.1775 |  
layer3.1.conv2  |  0.4382 | 0.1792 |  

#### Resutls: 
As the table2 in the paper, it indeed show that 
```
Second, different layers show different sensitivity to the quantization.  
For example, the downsampling convolution layers can be quantized much (a small α),
while some layers such as layer3.0.conv2 are not suitable for  quantization (a large α).  
```

#### Issue: 
It seems that α of weights is bigger than that of activations.  
Maybe the un-quantize batchnorm restricts the activation and cause the difference to the paper. (or someone can tell why)


### Update Note
> 20191218:
> Update uniform quantization results. It seems that the sgn function still need STE backward or the loss will becomes Nan.  
> 20191231:
> Update Experiments.
