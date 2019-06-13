import pathlib
import os
from PIL import Image
import numpy as np
import cv2 
import tensorflow as tf
import imutils #rotating images properly

def rotate(img):
    """
    Function that rotates an image 90 degrees 4 times.

    returns:
    4 image arrays each rotated 90 degrees
    """
    rotated90 = imutils.rotate_bound(img, 90)
    rotated180 = imutils.rotate_bound(img, 180)
    rotated270 = imutils.rotate_bound(img, 270)

    return img, rotated90, rotated180, rotated270

def downscale(img):
    """
    Downscales an image 0.9x, 0.8x, 0.7x and 0.6x.

    Returns:
    5 image arrays
    """
    (w, h) = img.shape[:2]
    img09 = cv2.resize(img, dsize=(int(h*0.9),int(w*0.9)), interpolation=cv2.INTER_CUBIC)
    img08 = cv2.resize(img, dsize=(int(h*0.8),int(w*0.8)), interpolation=cv2.INTER_CUBIC)
    img07 = cv2.resize(img, dsize=(int(h*0.7),int(w*0.7)), interpolation=cv2.INTER_CUBIC)
    img06 = cv2.resize(img, dsize=(int(h*0.6),int(w*0.6)), interpolation=cv2.INTER_CUBIC)

    return img, img09, img08, img07, img06

def augment(img):
    """
    Rotates and downscales an image. Creates 20x images.
    """
    augmented_images = []

    rotated_images = rotate(img)
    
    for img in rotated_images:
        downscaled_images = downscale(img)
        
        for im in downscaled_images:
            augmented_images.append(im)

    return augmented_images

def do_augmentations(path):
    """
    Does augmentations on all images in folder 'path'.
    """
    # get all image paths from folder
    dir = pathlib.Path(path)
    all_image_paths = list(dir.glob('*'))
    all_image_paths = [str(x) for x in all_image_paths]

    im_counter = 0
    # do augmentations
    for path in all_image_paths:
        # open current image as array
        img = Image.open(path)
        img = np.array(img)

        augm_counter = 0
        # get augmented images
        augmented_images = augment(img)
        for im in augmented_images: #save them all to ./augmented
            x = Image.fromarray(im)
            x.save("./augmented/img{}aug{}.png".format(im_counter, augm_counter))
            augm_counter += 1
        im_counter += 1


def make_dataset(upscale_factor):
    """
    Creates the superresolution patches placeholder dataset.
    """
    # make placeholders
    training_batches = tf.placeholder_with_default(tf.constant(32, dtype=tf.int64), shape=[], name="batch_size_input")
    path_inputs = tf.placeholder(tf.string, shape=[None])

    # make dataset
    path_dataset = tf.data.Dataset.from_tensor_slices(path_inputs)
    train_dataset = path_dataset.flat_map(lambda x: load_image(x, upscale_factor))
    train_dataset = train_dataset.shuffle(buffer_size=(91*20)) #91-image dataset times augmentations
    train_dataset = train_dataset.batch(training_batches)

    return train_dataset

def load_image(path, scale):
    """
    Loads an image into proper patches.
    """
    # init
    channels = 1
    lr_h, lr_w = 10, 10
    if(scale == 3):
        lr_h = 7
        lr_w = 7
    elif(scale == 4):
        lr_h = 6
        lr_w = 6

    #read tf image 
    im = tf.read_file(path)
    im = tf.image.decode_png(im, channels=3)
    im = tf.cast(im, tf.float32)

    # seperate rgb channels
    R, G, B = tf.unstack(im, 3, axis=2)

    # multiply by ?
    y = R * 0.299 + G * 0.587 + B * 0.114
    print("y.shape: ")
    print(y.shape)

    # shape to 1 channel and normalize
    im = tf.reshape(y, (tf.shape(im)[0], tf.shape(im)[1], 1)) / 255
    print("im.shape: ")
    print(im.shape)

    # make dimensions divisible by scale and make hr shape
    X = tf.dtypes.cast((tf.shape(im)[0] / scale), dtype=tf.int32) * scale
    Y = tf.dtypes.cast((tf.shape(im)[1] / scale), dtype=tf.int32) * scale
    high = tf.image.crop_to_bounding_box(im, 0, 0, X, Y)
    print("high.shape: ")
    print(high.shape)
    print("\n")

    # make lr shape
    imgshape = tf.shape(high)
    size = [imgshape[0] / scale, imgshape[1] / scale]
    low = tf.image.resize_images(high, size=size, method=tf.image.ResizeMethod.BILINEAR)
    print("low.shape: ")
    print(low.shape)
    print("\n")

    hshape = tf.shape(high)
    lshape = tf.shape(low)

    # make it 4d (1, h, w, channels)
    low_r = tf.reshape(low, [1, lshape[0], lshape[1], channels])
    high_r = tf.reshape(high, [1, hshape[0], hshape[1], channels])
    print("low_r.shape: ")
    print(low_r.shape)
    print("high_r.shape: ")
    print(high_r.shape)
    print("\n")

    # get image patches (size depending on scale)
    # ksizes = The size of the sliding window for each dimension of images.
    # strides = How far the centers of two consecutive patches are in the images.
    # rates = This is the input stride, specifying how far two consecutive patch samples are in the input.
    
    # TODO: SHOULD BE ONE PIXEL OVERLAPPING
    slice_l = tf.image.extract_image_patches(low_r, ksizes=[1, lr_h, lr_w, 1], strides=[1, lr_h, lr_w , 1], rates=[1, 1, 1, 1], padding="VALID")
    slice_h = tf.image.extract_image_patches(high_r, ksizes=[1, lr_h * scale, lr_w * scale, 1], strides=[1, lr_h * scale, lr_w * scale, 1],
                                                rates=[1, 1, 1, 1], padding="VALID")
    print("slice_l.shape: ")
    print(slice_l.shape)
    print("slice_h.shape: ")
    print(slice_h.shape)
    print("\n")

    #reshape patches to be in the shape (amount_of_patches, height, weight, channels)
    LR_image_patches = tf.reshape(slice_l, [tf.shape(slice_l)[1] * tf.shape(slice_l)[2], lr_h, lr_w, channels])
    HR_image_patches = tf.reshape(slice_h, [tf.shape(slice_h)[1] * tf.shape(slice_h)[2], lr_h * scale, lr_w * scale, channels])
    print("LR_image_patches.shape: ")
    print(LR_image_patches.shape)
    print("HR_image_patches.shape: ")
    print(HR_image_patches.shape)
    print("\n")

    return tf.data.Dataset.from_tensor_slices((LR_image_patches, HR_image_patches))
