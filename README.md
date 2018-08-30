# Repulsion Loss based on Faster R-CNN

## Introduction

This project is a repulsion loss implementation based on faster RCNN, aimed to recure the thesis "Repulsion loss" CVPR 2018. This project is based on the repo:
* [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), developed based on Pytorch

## Process

* change RPN scale to [3,6,9,12,15,18,21,24,27,30,33]

* dilation: remove the fouth maxpooling in vgg16, and add dilation in the next conv

* Ignore handling: 
lib/model/rpn/lib/model/rpn/anchor_target_layer.py 
lib/model/rpn/proposal_target_layer_cascade.py

* hard example: 
lib/datasets/pascal_voc.py change the label; 
lib/model/rpn/lib/model/rpn/anchor_target_layer.py 
lib/model/rpn/proposal_target_layer_cascade.py 

* reploss: 
lib/model/faster-rcnn/repulsion_loss.py

## Train 
``` python
python train_vgg_repulsion.py --cuda --mGPUs
```


