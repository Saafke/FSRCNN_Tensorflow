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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Start training the model.', action="store_true")
    parser.add_argument('--test', help='Run tests on the model', action="store_true")
    parser.add_argument('--load', help='Load previous model for training',action="store_true")
    parser.add_argument('--scale', default=2, help='Scaling factor of the model')
    parser.add_argument('--batch', type=int, help='Batch size of the training', default=1)
    parser.add_argument('--epochs', help='Number of epochs during training', default=20)
    parser.add_argument('--T91_dir', help='Path to 91-image dataset')
    parser.add_argument('--general100_dir', help='Path to General100 dataset')
    parser.add_argument('--d', help='Variable for d', default=56)
    parser.add_argument('--s', help='Variable for s', default=12)
    parser.add_argument('--m', help='Variable for m', default=4)

    args = parser.parse_args()

    # INIT
    scale = args.scale
    fsrcnn_params = (args.d, args.s, args.m) #d,s,m
    epochs = args.epochs
    batch = args.batch
    load_flag = args.load
    T91_dir = args.T91_dir
    general100_dir = args.general100_dir 
    
    learning_rate = 0.001

    # Set gpu 
    config = tf.ConfigProto() #log_device_placement=True
    config.gpu_options.allow_growth = True

    # Dynamic placeholders
    LR_holder = tf.placeholder(tf.float32, [None, None, None, 1], name='images')
    HR_holder = tf.placeholder(tf.float32, [None, None, None, 1], name='labels')

    # -- Model
    # construct model
    out, loss, train_op, psnr = fsrcnn.model(LR_holder, HR_holder, scale, batch, learning_rate, fsrcnn_params)

    # Create run instance
    run = run_utils.run(config, LR_holder, HR_holder)

    if args.train:
        # Augment and then load images
        utils.augment(T91_dir, save_path="./augmented")
        all_image_paths = utils.getpaths("./augmented")
        X_np, Y_np = utils.load_images(all_image_paths, scale)

        model_outputs = out, loss, train_op, psnr

        # Train
        run.train(X=X_np, Y=Y_np, epochs=epochs, batch=batch, load_flag=load_flag, model_outputs=model_outputs)

    if args.test:
        # Test image
        run.test(out)

    print("I ran successfully.")