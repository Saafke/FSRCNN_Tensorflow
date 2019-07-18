import tensorflow as tf 
import fsrcnn
import data_utils
import run
import os
import cv2
import numpy as np
import pathlib
import argparse
from PIL import Image
import numpy
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #gets rid of avx/fma warning

# TODO: 
# Overlapping patches
# seperate learning rate for deconv layer
# switch out deconv layer for different models
# train models for all different upscale factors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Train the model', action="store_true")
    parser.add_argument('--test', help='Run tests on the model', action="store_true")
    parser.add_argument('--export', help='Export the model as .pb', action="store_true")
    parser.add_argument('--fromscratch', help='Load previous model for training',action="store_false")
    parser.add_argument('--finetune', help='Finetune model on General100 dataset',action="store_true")
    parser.add_argument('--small', help='Run FSRCNN-small', action="store_true")
    parser.add_argument('--scale', type=int, help='Scaling factor of the model', default=2)
    parser.add_argument('--batch', type=int, help='Batch size of the training', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs during training', default=20)
    parser.add_argument('--image', help='Specify test image', default="./butterfly.png")
    parser.add_argument('--lr', type=float, help='Learning_rate', default=0.001)
    parser.add_argument('--d', type=int, help='Variable for d', default=56)
    parser.add_argument('--s', type=int, help='Variable for s', default=12)
    parser.add_argument('--m', type=int, help='Variable for m', default=4)
    parser.add_argument('--traindir', help='Path to train images')
    parser.add_argument('--finetunedir', help='Path to finetune images')
    parser.add_argument('--validdir', help='Path to validation images')

    args = parser.parse_args()

    # INIT
    scale = args.scale
    fsrcnn_params = (args.d, args.s, args.m) #d,s,m
    traindir = args.traindir

    augmented_path = "./augmented"
    small = args.small

    lr_size = 10
    if(scale == 3):
        lr_size = 7
    elif(scale == 4):
        lr_size = 6
        
    hr_size = lr_size * scale
    
    # FSRCNN-small
    if small:
        fsrcnn_params = (32, 5, 1)

    # Set checkpoint paths for different scales and models
    ckpt_path = ""
    if scale == 2:
        ckpt_path = "./CKPT_dir/x2/"
        if small:
            ckpt_path = "./CKPT_dir/x2_small/"
    elif scale == 3:
        ckpt_path = "./CKPT_dir/x3/"
        if small:
            ckpt_path = "./CKPT_dir/x3_small/"
    elif scale == 4:
        ckpt_path = "./CKPT_dir/x4/"
        if small:
            ckpt_path = "./CKPT_dir/x4_small/"
    else:
        print("Upscale factor scale is not supported. Choose 2, 3 or 4.")
        exit()
    
    # Set gpu 
    config = tf.ConfigProto() #log_device_placement=True
    config.gpu_options.allow_growth = True

    # Create run instance
    run = run.run(config, lr_size, ckpt_path, scale, args.batch, args.epochs, args.lr, args.fromscratch, fsrcnn_params, args.validdir)

    if args.train:
        # if finetune, load model and train on general100
        if args.finetune:
            traindir = args.finetunedir
            augmented_path = "./augmented_general100"

        # augment (if not done before) and then load images 
        data_utils.augment(traindir, save_path=augmented_path)

        run.train(augmented_path)

    if args.test:
        run.test(args.image)
        run.upscale(args.image)

    if args.export:
        run.export()
    
    print("I ran successfully.")