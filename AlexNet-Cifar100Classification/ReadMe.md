# AlexNet Example

## Description
Cifar100 classification using AlexNet

## DataSet
- [Download](http://www.cs.toronto.edu/~kriz/cifar.html)
- Pre-process
    - load data through pickle (see '1.4 pickle loader' in this [blog](https://blog.csdn.net/silence_iz/article/details/107604594))
    
- Images folder
```buildoutcfg
+images_folder
    +train
        +class_1
            image_1.jpg
            ...
         +class_2 
            image_1.jpg
            ...
    +val
        +class_1
            image_1.jpg
            ...
```

## Requirement
- Python >3.6
- PyTorch > 1.0
- CUDA
- TensorboardX

## Code
- alexnet_go.py
    - training process and saving checkpoints, note that you need to modify the image_folder to your images folder.
- alexnet_test.py
    - loading checkpoints adn testing process. 
- alexnet_model.py
    - alexnet structure.
- dataset.py
    - load data from images folder
- files.py
    - save and load checkpoints
- model_util.py
    - build net structure
    
## Usage
- training
```python
python alexnet_go.py
tensorboard --logdir ./runs
```
- testing
```buildoutcfg
python alexnet_test.py
```