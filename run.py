import tensorflow as tf
import os
import cv2
import numpy as np
import math
import data_utils
from skimage import io
import fsrcnn
from PIL import Image

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

class run:
    def __init__(self, config, lr_size, ckpt_path, scale, batch, epochs, lr, load_flag, fsrcnn_params, smallFlag, validdir):
        self.config = config
        self.lr_size = lr_size
        self.ckpt_path = ckpt_path
        self.scale = scale
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        self.load_flag = load_flag
        self.fsrcnn_params = fsrcnn_params
        self.smallFlag = smallFlag
        self.validdir = validdir
    
    def train(self, imagefolder):
        
        # Create training dataset iterator
        image_paths = data_utils.getpaths(imagefolder)
        train_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_dataset, 
                                                 output_types=(tf.float32, tf.float32), 
                                                 output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])),
                                                 args=[image_paths, self.scale])
        train_dataset = train_dataset.padded_batch(self.batch, padded_shapes=([None, None, 1],[None, None, 1]))
        
        # Create validation dataset
        val_image_paths = data_utils.getpaths(self.validdir)
        val_dataset = tf.data.Dataset.from_generator(generator=data_utils.make_val_dataset,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=(tf.TensorShape([None, None, 1]), tf.TensorShape([None, None, 1])),
                                                 args=[val_image_paths, self.scale])
        val_dataset = val_dataset.padded_batch(1, padded_shapes=([None, None, 1],[None, None, 1]))

        # Make the iterator and its initializers
        train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_initializer = train_val_iterator.make_initializer(train_dataset)
        val_initializer = train_val_iterator.make_initializer(val_dataset)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        LR, HR = iterator.get_next()

        if not self.smallFlag:
            print("\nRunning FSRCNN.")
        else:
            print("\nRunning FSRCNN-small.")
        out, loss, train_op, psnr = fsrcnn.model(LR, HR, self.lr_size, self.scale, self.batch, self.lr, self.fsrcnn_params)
        
        # -- Training session
        with tf.Session(config=self.config) as sess:
            
            train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
            sess.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver()
            
            # Create check points directory if not existed, and load previous model if specified.
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            else:
                if os.path.isfile(self.ckpt_path + "fsrcnn_ckpt" + ".meta"):
                    if self.load_flag:
                        saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))
                        print("Loaded checkpoint.")
                    if not self.load_flag:
                        print("No checkpoint loaded. Training from scratch.")
                else:
                    print("Previous checkpoint does not exists.")

            train_val_handle = sess.run(train_val_iterator.string_handle())

            print("Training...")
            for e in range(1, self.epochs+1):
                
                sess.run(train_initializer)
                step, train_loss, train_psnr = 0, 0, 0 

                while True:
                    try:
                        o, l, t, ps = sess.run([out, loss, train_op, psnr], feed_dict={handle: train_val_handle})
                        train_loss += l
                        train_psnr += (np.mean(np.asarray(ps)))
                        step += 1
                        
                        if step % 10000 == 0:
                            save_path = saver.save(sess, self.ckpt_path + "fsrcnn_ckpt")  
                            print("Step nr: [{}/{}] - Loss: {:.5f}".format(step, "?", float(train_loss/step)))
                    
                    except tf.errors.OutOfRangeError:
                        break

                # Perform end-of-epoch calculations here.
                sess.run(val_initializer)
                tot_val_psnr, val_im_cntr = 0, 0
                try:
                    while True:
                        val_psnr = sess.run([psnr], feed_dict={handle:train_val_handle})

                        tot_val_psnr += val_psnr[0]
                        val_im_cntr += 1

                except tf.errors.OutOfRangeError:
                    pass
                print("Epoch nr: [{}/{}]  - Loss: {:.5f} - val PSNR: {:.3f}\n".format(e,
                                                                                      self.epochs,
                                                                                      float(train_loss/step),
                                                                                      (tot_val_psnr[0] / val_im_cntr)))
                save_path = saver.save(sess, self.ckpt_path + "fsrcnn_ckpt")   

            print("Training finished.")
            train_writer.close()

    def upscale(self, path):
        """
        Upscales an image via model.
        """
        img = cv2.imread(path, 3)
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_y = img_ycc[:,:,0]
        floatimg = img_y.astype(np.float32) / 255.0
        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 1)

        with tf.Session(config=self.config) as sess:
            print("\nUpscale image by a factor of {}:\n".format(self.scale))
            
            # load and run
            ckpt_name = self.ckpt_path + "fsrcnn_ckpt" + ".meta"
            saver = tf.train.import_meta_graph(ckpt_name)
            saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))
            graph_def = sess.graph
            LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")
            HR_tensor = graph_def.get_tensor_by_name("NHWC_output:0")

            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})

            # post-process
            Y = output[0]
            Y = (Y * 255.0).clip(min=0, max=255)
            Y = (Y).astype(np.uint8)

            # Merge with Chrominance channels Cr/Cb
            Cr = np.expand_dims(cv2.resize(img_ycc[:,:,1], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC), axis=2)
            Cb = np.expand_dims(cv2.resize(img_ycc[:,:,2], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC), axis=2)
            HR_image = (cv2.cvtColor(np.concatenate((Y, Cr, Cb), axis=2), cv2.COLOR_YCrCb2BGR))

            bicubic_image = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

            cv2.imshow('Original image', img)
            cv2.imshow('HR image', HR_image)
            cv2.imshow('Bicubic HR image', bicubic_image)
            cv2.waitKey(0)
        sess.close()

    def test(self, path):
        """
        Test single image and calculate psnr.
        """
        fullimg = cv2.imread(path, 3)
        width = fullimg.shape[0]
        height = fullimg.shape[1]

        cropped = fullimg[0:(width - (width % self.scale)), 0:(height - (height % self.scale)), :]
        img = cv2.resize(cropped, None, fx=1. / self.scale, fy=1. / self.scale, interpolation=cv2.INTER_CUBIC)
        
        # to ycrcb and normalize
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_y = img_ycc[:,:,0]
        floatimg = img_y.astype(np.float32) / 255.0
        
        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 1)

        with tf.Session(config=self.config) as sess:
            print("\nTest model with psnr:\n")
            # load the model
            ckpt_name = self.ckpt_path + "fsrcnn_ckpt" + ".meta"
            saver = tf.train.import_meta_graph(ckpt_name)
            saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))
            graph_def = sess.graph
            LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")
            HR_tensor = graph_def.get_tensor_by_name("NHWC_output:0")

            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})

            # post-process
            Y = output[0]
            Y = (Y * 255.0).clip(min=0, max=255)
            Y = (Y).astype(np.uint8)

            # Merge with Chrominance channels Cr/Cb
            Cr = np.expand_dims(cv2.resize(img_ycc[:,:,1], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC), axis=2)
            Cb = np.expand_dims(cv2.resize(img_ycc[:,:,2], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC), axis=2)
            HR_image = (cv2.cvtColor(np.concatenate((Y, Cr, Cb), axis=2), cv2.COLOR_YCrCb2BGR))

            bicubic_image = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

            print("PSNR of fsrcnn  upscaled image: {}".format(self.psnr(cropped, HR_image)))
            print("PSNR of bicubic upscaled image: {}".format(self.psnr(cropped, bicubic_image)))

            cv2.imshow('Original image', fullimg)
            cv2.imshow('HR image', HR_image)
            cv2.imshow('Bicubic HR image', bicubic_image)
            
            cv2.imwrite("./images/fsrcnnOutput.png", HR_image)
            cv2.imwrite("./images/bicubicOutput.png", bicubic_image)
            cv2.imwrite("./images/original.png", fullimg)
            cv2.imwrite("./images/input.png", img)
            
            cv2.waitKey(0)
        sess.close()
        
    def load_pb(self, path_to_pb):
        with tf.gfile.GFile(path_to_pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            return graph

    def testFromPb(self, path):
        # Read model
        pbPath = "./models/FSRCNN_x{}.pb".format(self.scale)
        if self.smallFlag:
            pbPath = "./models/FSRCNN-small_x{}.pb".format(self.scale)

        # Get graph
        graph = self.load_pb(pbPath)

        fullimg = cv2.imread(path, 3)
        width = fullimg.shape[0]
        height = fullimg.shape[1]

        cropped = fullimg[0:(width - (width % self.scale)), 0:(height - (height % self.scale)), :]
        img = cv2.resize(cropped, None, fx=1. / self.scale, fy=1. / self.scale, interpolation=cv2.INTER_CUBIC)
        
        # to ycrcb and normalize
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_y = img_ycc[:,:,0]
        floatimg = img_y.astype(np.float32) / 255.0
        
        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 1)

        LR_tensor = graph.get_tensor_by_name("IteratorGetNext:0")
        HR_tensor = graph.get_tensor_by_name("NHWC_output:0")

        with tf.Session(graph=graph) as sess:
            print("Loading pb...")
            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})
            
            # post-process
            Y = output[0]
            Y = (Y * 255.0).clip(min=0, max=255)
            Y = (Y).astype(np.uint8)

            # Merge with Chrominance channels Cr/Cb
            Cr = np.expand_dims(cv2.resize(img_ycc[:,:,1], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC), axis=2)
            Cb = np.expand_dims(cv2.resize(img_ycc[:,:,2], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC), axis=2)
            HR_image = (cv2.cvtColor(np.concatenate((Y, Cr, Cb), axis=2), cv2.COLOR_YCrCb2BGR))

            bicubic_image = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

            print("PSNR of FSRCNN  upscaled image: {}".format(self.psnr(cropped, HR_image)))
            print("PSNR of bicubic upscaled image: {}".format(self.psnr(cropped, bicubic_image)))

            cv2.imshow('Original image', fullimg)
            cv2.imshow('FSRCNN upscaled image', HR_image)
            cv2.imshow('Bicubic upscaled image', bicubic_image)

            cv2.imwrite("./images/fsrcnnOutput.png", HR_image)
            cv2.imwrite("./images/bicubicOutput.png", bicubic_image)
            cv2.imwrite("./images/original.png", fullimg)
            cv2.imwrite("./images/input.png", img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        sess.close()
    
    def export(self):

        if not os.path.exists("./models"):
            os.makedirs("./models")

        print("Exporting model...")

        graph = tf.get_default_graph()
        with graph.as_default():
            with tf.Session(config=self.config) as sess:
                
                ### Restore checkpoint
                ckpt_name = self.ckpt_path + "fsrcnn_ckpt" + ".meta"
                saver = tf.train.import_meta_graph(ckpt_name)
                saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_path))

                # Return a serialized GraphDef representation of this graph
                graph_def = sess.graph.as_graph_def()

                # All variables to constants
                graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['NCHW_output'])
                
                # Optimize for inference
                graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ["IteratorGetNext"],
                                                                            ["NCHW_output"],  # ["NHWC_output"],
                                                                            tf.float32.as_datatype_enum)

                graph_def = TransformGraph(graph_def, ["IteratorGetNext"], ["NCHW_output"], ["sort_by_execution_order"])

                pb_filename = "./models/FSRCNN_x{}.pb".format(self.scale)
                if self.smallFlag:
                    pb_filename = "./models/FSRCNN-small_x{}.pb".format(self.scale)
                
                with tf.gfile.FastGFile(pb_filename, 'wb') as f:
                    f.write(graph_def.SerializeToString())

                tf.train.write_graph(graph_def, ".", 'train.pbtxt')

    def psnr(self, img1, img2):
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))