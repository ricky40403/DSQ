# PyTranformer



## summary
This repository implement the summary function similar to keras summary()  

```
model = nn.Sequential(
          nn.Conv2d(3,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

model.eval()

transofrmer = TorchTransformer()
input_tensor = torch.randn([1, 3, 224, 224])
net = transofrmer.summary(model, input_tensor)

##########################################################################################
Index| Layer (type)    | Bottoms         Output Shape              Param #
---------------------------------------------------------------------------
    1| Data            |                 [(1, 3, 224, 224)]        0
---------------------------------------------------------------------------
    2| Conv2d_1        | Data            [(1, 20, 220, 220)]       1500
---------------------------------------------------------------------------
    3| ReLU_2          | Conv2d_1        [(1, 20, 220, 220)]       0
---------------------------------------------------------------------------
    4| Conv2d_3        | ReLU_2          [(1, 64, 216, 216)]       32000
---------------------------------------------------------------------------
    5| ReLU_4          | Conv2d_3        [(1, 64, 216, 216)]       0
---------------------------------------------------------------------------
==================================================================================
Total Trainable params: 33500
Total Non-Trainable params: 0
Total params: 33500
```  

other  example is in [example.ipynb](summary_example.ipynb)

## visualize
visualize using [graphviz](https://graphviz.readthedocs.io/en/stable/) and [pydot](https://pypi.org/project/pydot/)  
it will show the architecture.  
Such as alexnet in torchvision:
```
model = models.__dict__["alexnet"]()
model.eval()
transofrmer = TorchTransformer()
transofrmer.visualize(model, save_name= "example", graph_size = 80)
# graph_size can modify to change the size of the output graph
# graphviz does not auto fit the model's layers, which mean if the model is too deep.
# And it will become too small to see.
# So change the graph size to enlarge the image for higher resolution.
```  
<img src=/examples/alexnet.png  height =800  width=100> 

example is in [example](visualize_example.ipynb)  
other example image is in [examples](/examples)

## transform layers
you can register layer type to transform  
First you need to register to transformer and the transformer will transform layers you registered. 

example in in [transform_example](transform_example.ipynb)




## Note
Suggest that the layers input should not be too many because the graphviz may generate image slow.(eg: densenet161 in torchvision 0.4.0 may stuck when generating png)

## TODO
- [x] support registration(replace) for custom layertype
- [ ] support replacement of specified layer in model for specified layer
- [x] activation size calculation for supported layers
- [x] network summary output as in keras
- [x] model graph visualization
