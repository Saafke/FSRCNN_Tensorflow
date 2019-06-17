import tensorflow as tf 
import fsrcnn
import utils
import os
import cv2
import numpy as np
import pathlib
from PIL import Image
import numpy
from tensorflow.python.client import device_lib

def split(image,nrows,ncols):
    n, m = image.shape
    return (image.reshape(n // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))

def train():
    # -- Training session
    with tf.Session(config=config) as sess:
        #merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        # create check points directory
        if not os.path.exists("./CKPT_dir"):
            os.makedirs("./CKPT_dir")
        else:
            if os.path.isfile("./CKPT_dir/fsrcnn_ckpt" + ".meta") and load_flag:
                saver.restore(sess, tf.train.latest_checkpoint("./CKPT_dir"))
                print("Loaded checkpoint.")
            else:
                print("Previous checkpoint does not exists.")

        for e in range(epochs):
            sess.run(iter.initializer, feed_dict={path_inputs: all_image_paths, training_batches: batch})
            train_loss, train_psnr = 0, 0
            #sess.run(next)
            count = 0
            try:
                while count <= 1820:
                    o, l, t, ps = sess.run([out, loss, train_op, psnr])
                    train_loss += l
                    train_psnr += ps
                    count += 1

                print(o * 255)
                #counting up losses
                print("Epoch no:", '%03d' % (e), "avg_loss:","{:.5f}".format(float(train_loss / 1820)),
                                                 "avg_psnr:","{:.3f}".format(float(train_psnr / 1820)))
                # save
                save_path = saver.save(sess, "./CKPT_dir/fsrcnn_ckpt")
            except tf.errors.OutOfRangeError:
                pass

        train_writer.close()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #gets rid of avx/fma warning
#print("This is opencv version: ", cv2.__version__)
#print(device_lib.list_local_devices())
#print("\n")
#print(tf.test.gpu_device_name())
#print("\n")

# INIT
scale = 2
fsrcnn_params = (56,12,4) #d,s,m
learning_rate = 0.001
epochs = 10
batch = 1
load_flag = False

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
config = tf.ConfigProto() #log_device_placement=True
config.gpu_options.allow_growth = True

# make the tensorflow dataset and iterator
path_inputs, training_batches, train_dataset = utils.make_dataset(scale, batch)
iter = train_dataset.make_initializable_iterator()
LR, HR = iter.get_next()
next = iter.get_next()

# -- Model
# construct model
out, loss, train_op, psnr = fsrcnn.model(LR, HR, scale, batch, learning_rate, fsrcnn_params)

print("Training...")
#train()
print("Training finished")

# -- Testing stuff ----------------------------------
print("Not implemented yet.")

scale = 2
lr_size = 10
hr_size = 10 * scale

# get patches of test image
test_image_path = "./butterfly.png"
LR_patches, HR_patches = utils.load_image(test_image_path, 2)

#---------------------
# get patches via numpy shizzle
np_im = cv2.imread("/home/weber/Documents/gsoc/FSRCNN/FSRCNN_Tensorflow/augmented/img85aug0.png", 3)
np_im = cv2.imread("./butterfly.png", 3)
print("IMPROTANT:", np_im.shape)
upsampled_im = cv2.resize(np_im, (np_im.shape[1]*2, np_im.shape[0]*2), interpolation=cv2.INTER_CUBIC)
print("IMPROTANT22222:", upsampled_im.shape)
#np_im = cv2.cvtColor(np_im, cv2.COLOR_BGR2YCrCb)

# get patch
np_im = np_im[0:20,0:20,:]
print("numpy shape", np_im.shape)

cv2.namedWindow('LR_patch', cv2.WINDOW_NORMAL)
cv2.imshow('LR_patch', np_im)
cv2.waitKey(0)

R_orig = upsampled_im[0:20,0:20,0]
G_orig = upsampled_im[0:20,0:20,1]
B_orig = upsampled_im[0:20,0:20,2]

# unstack
R = (np_im[:,:,0] * 0.299) / 255
G = (np_im[:,:,1] * 0.587) / 255
B = (np_im[:,:,2] * 0.114) / 255

R = np.expand_dims(R,0)
R = np.expand_dims(R,3)
R = tf.convert_to_tensor(R)

G = np.expand_dims(G,0)
G = np.expand_dims(G,3)
G = tf.convert_to_tensor(G)

B = np.expand_dims(B,0)
B = np.expand_dims(B,3)
B = tf.convert_to_tensor(B)

print("R:", R.shape)
print("G:", G.shape)
print("B:", B.shape)

#---------------------------------

# just take one patch
single_lr_image_patch = LR_patches[0:1,:,:,:]
#single_lr_image_patch = tf.convert_to_tensor(single_lr_image_patch)
print("Single lr_image patch shape:", single_lr_image_patch.shape)


print("Start running tests on the model")
graph = tf.get_default_graph()
with graph.as_default():
    with tf.Session(config=config) as sess:

        ### Restore checkpoint
        ckpt_name = "./CKPT_dir/fsrcnn_ckpt" + ".meta"
        saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.import_meta_graph(ckpt_name)
        saver.restore(sess, tf.train.latest_checkpoint("./CKPT_dir"))
        print("Loaded model.")

        ### Get input and output nodes
        LR_tensor = graph.get_tensor_by_name("IteratorGetNext:0")
        HR_tensor = graph.get_tensor_by_name("NHWC_output:0")

        ### Set input
        LR_input = single_lr_image_patch
        LR_input1 = sess.run(R) 
        LR_input2 = sess.run(G) 
        LR_input3 = sess.run(B) 
        
        print("before inference")
        ### Run inference
        opt1 = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input1})
        opt2 = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input2})
        opt3 = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input3})
        print("after inference:", opt1.shape)
        print(opt1.shape)
        opt1 = numpy.squeeze(opt1) * 255 * (1/0.299)
        opt1 = opt1.astype(np.uint8)

        opt2 = numpy.squeeze(opt2) * 255 * (1/0.587)
        opt2 = opt2.astype(np.uint8)

        opt3 = numpy.squeeze(opt3) * 255 * (1/0.114)
        opt3 = opt3.astype(np.uint8)
        print(opt3.dtype)

        print(R_orig.dtype)

        print("MSE:", np.square(np.subtract(opt1, R_orig)).mean())
        HR_img = numpy.stack((opt1, G_orig,B_orig),2)
        print("hrimg shape", HR_img.shape)

        cv2.namedWindow('HR patch', cv2.WINDOW_NORMAL)
        cv2.imshow('HR patch', HR_img)
        cv2.waitKey(0)

        HR_img_bicubic = numpy.stack((R_orig,G_orig,B_orig),2)

        cv2.namedWindow('HR patch bicubic', cv2.WINDOW_NORMAL)
        cv2.imshow('HR patch bicubic', HR_img_bicubic)
        cv2.waitKey(0)


print("I ran successfully.")