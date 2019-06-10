import cv2
import tensorflow as tf 
import numpy as np 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #gets rid of avx/fma warning

print("This is opencv version: ", cv2.__version__)

tf.enable_eager_execution()
print(tf.reduce_sum(tf.random_normal([1000, 1000])))


