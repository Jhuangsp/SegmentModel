
        
import tensorflow as tf
import os
import numpy as np

class ResNet(object):
    def __init__(self, name='resnet', layer_n=3):
        super(ResNet, self).__init__()

        def preproc(x):
            mean = tf.reduce_mean(x, axis=1, keepdims=True)
            return x - mean

        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, 20, 18, 2], name='X') # [batch, height, width, channel]
            self.y = tf.placeholder(tf.float32, [None, 10], name='y') # [batch, width(frames)]
            # self.training = tf.placeholder(tf.bool, name='training')

            x = preproc(self.X)
            self.logits = self.build_net(x_img=x, layer_n=layer_n)

    def residual_block(self, x, o_channel, downsampling=False, name='res_block'):
        '''
            residual_block_v1
            kernel size:
                [3,3]
                [3,3]
        '''
        # input_channel = int(x.shape[-1]) # get # of input channels

        if downsampling:
            stride = 2
        else:
            stride = 1

        with tf.variable_scope(name):
            with tf.variable_scope('conv1_in_block'):
                h1 = tf.layers.conv2d(x, o_channel, kernel_size=[3,3], strides=stride, padding='SAME')
                h1 = tf.layers.batch_normalization(h1)
                h1 = tf.nn.relu(h1)
            
            with tf.variable_scope('conv2_in_block'):
                h2 = tf.layers.conv2d(h1, o_channel, kernel_size=[3,3], strides=1, padding='SAME')
                h2 = tf.layers.batch_normalization(h2)

            if downsampling:
                x = tf.layers.conv2d(x, o_channel, kernel_size=[1,1], strides=stride, padding='SAME')

            return tf.nn.relu(h2 + x)

    def build_net(self, x_img, layer_n):
        x = x_img
        o_frames = x_img.shape[1] // 2
        ori_shape = x_img.shape[1:-1]

        with tf.variable_scope("conv0"): # init conv layer
            x = tf.layers.conv2d(x, filters=16, kernel_size=[3,3], strides=1, padding='SAME')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)

        with tf.variable_scope("conv1"):
            for i in range(layer_n):
                x = self.residual_block(x, o_channel=16, name="resblock{}".format(i+1))
                assert (np.array(x.shape[1:-1]) == np.array(x_img.shape[1:-1])).all(), '{}, {}'.format(np.array(x.shape[1:-1]), np.array(x_img.shape[1:-1]))

        with tf.variable_scope("conv2"):
            for i in range(layer_n):
                x = self.residual_block(x, o_channel=32, downsampling=(i==0), name="resblock{}".format(i+1))
                assert (np.array(x.shape[1:-1]) == np.array(x_img.shape[1:-1]) // 2).all(), '{}, {}'.format(np.array(x.shape[1:-1]), np.array(x_img.shape[1:-1]))

        # with tf.variable_scope("conv3"):
        #     for i in range(layer_n):
        #         x = self.residual_block(x, o_channel=64, downsampling=(i==0), name="resblock{}".format(i+1))
        #         assert x.shape[1:-1] == ori_shape // 4

        with tf.variable_scope("fc"):
            x = tf.reduce_mean(x, axis=[1,2]) # global average pooling
            assert x.shape[1:] == [32], '{}'.format(x.shape)

            logits = tf.sigmoid(tf.layers.dense(x, o_frames, name="logits"))

        return logits
