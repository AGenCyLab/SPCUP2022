#!/bin/bash

python train_cnn.py --gpu-indices=0,1,2,3,4,5,6,7 --num-workers=8 --model-type=ResNet18 --load-eval-data=0 --epochs=200 --checkpoint-path=./resnet18_on_train_dataset
