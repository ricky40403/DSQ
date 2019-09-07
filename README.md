# DSQ
pytorch implementation of "Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks"  

****
The Origin Paper : <https://arxiv.org/abs/1908.05033>  
****

This repository follow the Algorithm 1 in the paper.  
As mention in the paper: 
```
For clipping value l and u, we try the following two strategies:  
moving average statistics and optimization by backward propagation.
```
Because the paper fine-tunes from the pre-trained model, so it can find the clipping value.  
Instead of using the strategies, this repository uses the max value of int32 as the initial value.  
It should not affect the value range (because the parameter of the deep model should not too large), and most of the edge device range is up to int32.  


