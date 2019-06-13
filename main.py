import tensorflow as tf 
import fsrcnn
import utils
import os
import cv2
import pathlib
from PIL import Image
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #gets rid of avx/fma warning
print("This is opencv version: ", cv2.__version__)

# INIT
scale = 2
fsrcnn_params = (56,12,4) #d,s,m
learning_rate = 0.001
epochs = 10
batch = 32

# -- Data
# augment images
if(not os.path.isdir("./augmented")):
    print("Making augmented images...")
    os.mkdir("./augmented")

    utils.do_augmentations("/home/weber/Documents/gsoc/datasets/T91")
    
    #count new images
    path, dirs, files = next(os.walk("./augmented"))
    file_count = len(files)
    print("{} augmented images are stored in the folder ./augmented".format(file_count))

# get all augmented images paths
DATA = pathlib.Path("./augmented")
all_image_paths = list(DATA.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]

# set gpu 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# make the tensorflow dataset and iterator
train_dataset = utils.make_dataset(scale)
iter = train_dataset.make_initializable_iterator()
LR, HR = iter.get_next()
next = iter.get_next()

# -- Model
# construct model
loss, train_op, psnr = fsrcnn.model(LR, HR, scale, fsrcnn_params)

# -- Training session
with tf.Session(config=config) as sess:
    #merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    # create check points directory
    if not os.path.exists("./CKPT_dir"):
        os.makedirs("./CKPT_dir")
    # else:
    #     if os.path.isfile(ARGS["CKPT"] + ".meta"):
    #         saver.restore(sess, tf.train.latest_checkpoint(ARGS["CKPT_dir"]))
    #         print("Loaded checkpoint.")
    #     else:
    #         print("Previous checkpoint does not exists.")

    for e in range(epochs):
        sess.run(iter.initializer, feed_dict={path_inputs: all_image_paths, training_batches: batch})
        count = 0
        sess.run(next)
        try:
            while True:
                l, t, ps = sess.run([loss, train_op, psnr])
                count = count + 1
                if count % 200 == 0:
                    print("Data count:", '%04d' % (count + 1), "Epoch no:", '%04d' % (e + 1), "loss:","{:.9f}".format(l))
                if count % 1000 == 0:
                    save_path = saver.save(sess, "./CKPT")
                    print("Model saved in path: %s" % save_path)
        except tf.errors.OutOfRangeError:
            pass

    train_writer.close()

print("I ran successfully.")