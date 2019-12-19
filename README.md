# DSQ
pytorch implementation of "Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks"  


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
-quantize_input : quantize input or not  
--quan_bit : quantization bit num  
--log_path : tensorboard log path to write, default folder is ./log.  

Examples  
Training DSQ with 8 bit (no quantiza input)
```
python train.py -a resnet18  -q DSQ --quan_bit 8
```
Training DSQ with 8 bit ( quantiza input)
```
python train.py -a resnet18  -q DSQ --quan_bit 8 -quantize_input
```

# Experiments

<table>
<tr><th> model </th> <th> QuanType </th> <th> W/A bit </th> <th> top1 </th> <th> top5 </th></tr>  
<tr><th rowspan="4"> resnet18 </th> <th> UniformQuan </th> <th> 4/32 </th> <th> 69.372 </th> <th> 88.824 </th></tr>
<tr><th> DSQ </th> <th> 4/32 </th> <th>  </th> <th>  </th></tr>
<tr><th> UniformQuan </th> <th> 4/4 </th> <th>  </th> <th>  </th></tr>
<tr><th> DSQ </th> <th> 4/4 </th> <th>  </th> <th>  </th></tr>
</table>

  


### Update Note
> 20191218:
> Update uniform quantization results. It seems that the sgn function still need STE backward or the loss will becomes Nan.
