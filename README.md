# FSRCNN_Tensorflow
Tensorflow implementation of [FSRCNN](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html) [1].

Still under development...

## Dependencies
- Tensorflow
- Python
- Numpy
- cv2
- imutils

## How to run

###### training:
`python main.py --T91_dir <T91-image dataset directory> --train --fromscratch`

This will create the T91 augmented dataset and train from scratch.

`python main.py --train --fromscratch --small`

This will create a FSRCNN-small network and train from scratch.

###### finetuning:
`python main.py --general100_dir <general100 dataset directory> --train --finetune`

This will 
- load the trained model
- create augmented general100 dataset (if it not exists yet)
- resume training but now on the augmented general100 dataset

###### testing:
`python main.py --test`

###### exporting file to .pb format:
`python3 main.py --export`

###### extra arguments (different scale, batch-sizes etc.)
`python main.py --h`

## Example
(1) Original picture
(2) Input image
(3) Bicubic scaled (3x) image
(4) ESPCN scaled (3x) image
![Alt text](images/original.png?raw=true "Original picture")
![Alt text](images/input.png?raw=true "Input image picture")
![Alt text](images/bicubicOutput.png?raw=true "Bicubic picture")
![Alt text](images/fsrcnnOutput.png?raw=true "FSRCNN picture")

## References
1. Chao Dong, Chen Change Loy, Xiaoou Tang. Accelerating the Super-Resolution Convolutional Neural Network, in Proceedings of European Conference on Computer Vision (ECCV), 2016