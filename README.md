# FSRCNN_Tensorflow
TensorFlow implementation of [Accelerating the Super-Resolution Convolutional Neural Network](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html) [1].

This implementation replaces the transpose conv2d layer by a sub-pixel layer [2]. 

Includes pretrained models for scales x2, x3 and x4. Which were trained on T91-image dataset, and finetuned on General100 dataset.

## Google Summer of Code with OpenCV
This repository was made during the 2019 GSoC program for the organization OpenCV. The [trained models (.pb files)](https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models/) in this repo can easily be used for inference in OpenCV with the 'dnn_superres' module. See the OpenCV documentation for how to do this. (OpenCV Code not yet merged, so please wait for this feature.)

## Requirements
- Tensorflow
- Python
- Numpy
- cv2
- imutils

# Running

Download [T91, General100 and Set14](http://vllab.ucmerced.edu/wlai24/LapSRN/).

T91 is for training. General100 for finetuning. Set14 as the validation set.

Train:
- from scratch\
`python main.py --train --scale <scale> --fromscratch --traindir /path-to-train_images/ --validdir /path-to-valid_images/`

- load previous\
`python main.py --train --scale <scale> --traindir /path-to-train_images/ --validdir /path-to-valid_images/`

- finetune\
`python main.py --train --scale <scale> --finetune --finetunedir /path-to-images/ --validdir /path-to-valid_images/`

Test:\
`python main.py --test --image /image-path/`

Export:\
`python3 main.py --export`

Extra arguments (Fsrcnn small, batch size, lr etc.):\
`python main.py --h`

## Example
(1) Original picture\
(2) Input image\
(3) Bicubic scaled (3x) image\
(4) FSRCNN scaled (3x) image\
![Alt text](images/original.png?raw=true "Original picture")
![Alt text](images/input.png?raw=true "Input image picture")
![Alt text](images/bicubicOutput.png?raw=true "Bicubic picture")
![Alt text](images/fsrcnnOutput.png?raw=true "FSRCNN picture")

## Notes
FSRCNN-small is a network with fewer parameters. Thus it is faster but has lower performance.

## References
1. Chao Dong, Chen Change Loy, Xiaoou Tang. [Accelerating the Super-Resolution Convolutional Neural Network](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html), in Proceedings of European Conference on Computer Vision (ECCV), 2016
2. [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158). By Shi et. al.  