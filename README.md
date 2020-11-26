# VRDL Homework 2
Code for mAP 0.44387 solution in VRDL Homework 2

## Abstract
In this work, I use EfficientDet I edited as my model<br>
I changed the input image size from (512, 512) to (128, 128) and changed EfficientNet-b0 to ResNet34<br>
Also, My feature pyramid has only 4 feature maps with shape (64, 64), (32, 32), (16, 16), (8, 8) 

## Reference
EfficientDet [Paper](https://arxiv.org/pdf/1911.09070.pdf ) | [GitHub](https://github.com/signatrix/efficientdet)

## Hardware
The following specs were used to create the solutions.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- 3x GeForce RTX 2080 Ti

## Reproducing Submission
To reproduct my submission, do the following steps:
1. [Installation](#installation)
2. [Prepare Data](#dataset-preparation)
3. [Download Pretrained models](#pretrained-models)
4. [Inference](#inference)

## Producing Your Own Submission
To produce your own submission, do the following steps:
1. [Installation](#installation)
2. [Prepare Data](#dataset-preparation)
3. [Train and Make Submission](#train-and-make-prediction)

## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is recommended.
```
virtualenv .
source bin/activate
pip3 install -r requirements.txt
```

## Dataset Preparation
You need to download the data [here](https://drive.google.com/drive/u/1/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl) by yourself.<br>
Unzip it and put them into the same directory below
```
VRDL_HW2
  +- data
  | + train
  | | + 1.png
  | | + ...
  | | + 33402.png
  | | + digitStruct.mat
  | + test
  | | + 1.png
  | | + ...
  | | + 13068.png
  +- src
  +- predict.py
  +- readfile.py
  +- train.py
  +- util.py
```
And run the following command to prepare annotations of training data and validation data
```
python3 prepare.py
```
## Pretrained models
You can download pre-trained model that used for my submission from this [link](https://drive.google.com/file/d/128f_l55fRxIXO-HkBsqUppC_a-ONF5Lg/view?usp=sharing)<br>
And put it into the following directory:
```
VRDL_HW2
  +- data
  +- EXP
  | +- HW2model.pth.tar
  +- src
  +- predict.py
  +- readfile.py
  +- train.py
  +- util.py
```

## Inference
Run the following command to reproduct my prediction.
```
python3 predict.py
```
It will generate a file named submission.json and it is my prediction whose mAP is 0.44387


## Train and Make Prediction
You can simply run the following command to train your models and make submission.
```
$ python train.py --expname={your experiment's name}
```

The expected training time is:

GPUs | Training Epochs | Training Time
------------- | ------------- | ------------- 
3x 2080 Ti | 200 | 13 hours

After finishing training your model, run the following command to make your prediction
```
python3 predict.py --expname={your experiment's name}
```
It will generate a file submission.json which is the prediction of the testing dataset<br>
Use this json file to make your submission!

## Citation
```
@article{EfficientDetSignatrix,
    Author = {Signatrix GmbH},
    Title = {A Pytorch Implementation of EfficientDet Object Detection},
    Journal = {https://github.com/signatrix/efficientdet},
    Year = {2020}
}
```
