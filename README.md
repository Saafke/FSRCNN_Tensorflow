# FSRCNN_Tensorflow
Tensorflow implementation of FSRCNN
http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html

Still under development...

## Dependencies
- Tensorflow
- Python
- Numpy
- cv2
- imutils

## How to run

###### How to train
`python main.py --T91_dir <T91-image dataset directory> --train`

This will create the T91 augmented dataset and train on it.

###### How to finetune
`python main.py --general100_dir <general100 dataset directory> --train --load --finetune`

This will 
- load the trained model
- create augmented general100 dataset (if it not exists yet)
- resume training on the augmented general100 dataset

###### How to test
`python main.py --test`

###### Extra arguments
For extra arguments run:
`python main.py --h`