import tensorflow as tf 
import fsrcnn
import utils
import run_utils
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
# [DONE] prelu
# [DONE] finetune on general100 
# [DONE] add argument for fsrcnn-small
# [DONE] Fix white/black specks
# [DONE] Provide testcode for comparing with bilinear/bicubic
# [DONE] Function to export model to .pb file
# [DONE] Move to this project's master branch
# Fix tf.shape so it can export properly
# Overlapping patches 
# Remove old code
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
    parser.add_argument('--general100_dir', help='Path to General100 dataset')

    args = parser.parse_args()

    # INIT
    scale = args.scale
    fsrcnn_params = (args.d, args.s, args.m) #d,s,m
    epochs = args.epochs
    batch = args.batch
    finetune = args.finetune
    learning_rate = args.lr
    load_flag = args.fromscratch
    traindir = args.traindir
    general100_dir = args.general100_dir 
    test_image = args.image

    dataset_path = traindir
    augmented_path = "./augmented"
    small = args.small

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

    # Dynamic placeholders
    LR_holder = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
    HR_holder = tf.placeholder(tf.float32, [None, None, None, 1], name='labels')
    HR_holder_shape = tf.shape(HR_holder)
    
    # -- Model
    # construct model
    out, loss, train_op, psnr = fsrcnn.model(LR_holder, HR_holder, HR_holder_shape, scale, batch, learning_rate, fsrcnn_params)

    # Create run instance
    run = run_utils.run(config, ckpt_path, LR_holder, HR_holder)

    if args.train:
        # If finetune, load model and train on general100
        if finetune:
            dataset_path = general100_dir
            augmented_path = "./augmented_general100"

        # Augment and then load images
        utils.augment(dataset_path, save_path=augmented_path)
        all_image_paths = utils.getpaths(augmented_path)
        X_np, Y_np = utils.load_images(all_image_paths, scale)

        model_outputs = out, loss, train_op, psnr

        # Train
        run.train(X=X_np, Y=Y_np, epochs=epochs, batch=batch, load_flag=load_flag, model_outputs=model_outputs)

    if args.test:
        # Test image
        run.test_compare(test_image, out, scale)

    if args.export:
        run.export(scale)
    print("I ran successfully.")