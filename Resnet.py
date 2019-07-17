
        
import tensorflow as tf
import os
import numpy as np

class ResNet(object):
    def __init__(self, name='resnet', layer_n=3, in_shape=[20, 18, 2], out_shape=[10]):
        super(ResNet, self).__init__()

        def preproc(x):
            mean = tf.reduce_mean(x, axis=1, keepdims=True)
            return x - mean

        self.out_shape = out_shape

        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, [None, in_shape[0], in_shape[1], in_shape[2]], name='X') # [batch, height, width, channel]
            self.y = tf.placeholder(tf.float32, [None, out_shape[0]], name='y') # [batch, width(frames)]

            x = preproc(self.X)
            self.logits = self.build_net(x_img=x, layer_n=layer_n)

    def residual_block(self, x, o_channel, downsampling=False, name='res_block'):
        '''
            residual_block_v1
            kernel size:
                [3,3]
                [3,3]
        '''
        stride = 2 if downsampling else 1

        with tf.variable_scope(name):
            with tf.variable_scope('conv1_in_block'):
                h = tf.layers.conv2d(x, o_channel, kernel_size=[3,3], strides=stride, padding='SAME')
                h = tf.layers.batch_normalization(h)
                h = tf.nn.elu(h)
            
            with tf.variable_scope('conv2_in_block'):
                h = tf.layers.conv2d(h, o_channel, kernel_size=[3,3], strides=1, padding='SAME')
                h = tf.layers.batch_normalization(h)

            if downsampling:
                x = tf.layers.conv2d(x, o_channel, kernel_size=[1,1], strides=stride, padding='SAME')

            return tf.nn.elu(h + x)

    def residual_block_v2(self, x, o_channel, downsampling=False, name='res_block'):
        '''
            residual_block_v2
            kernel size:
                [1,1] out=64
                [3,3] out=64
                [1,1] out=256

                [1,1] out=128
                [3,3] out=128
                [1,1] out=512
        '''
        stride = 2 if downsampling else 1

        with tf.variable_scope(name):

            with tf.variable_scope('conv1_in_block'):
                h = tf.layers.conv2d(x, o_channel//4, kernel_size=[1,1], strides=stride, padding='SAME')
                h = tf.layers.batch_normalization(h)
                h = tf.nn.elu(h)
            
            with tf.variable_scope('conv2_in_block'):
                h = tf.layers.conv2d(h, o_channel//4, kernel_size=[3,3], strides=1, padding='SAME')
                h = tf.layers.batch_normalization(h)
                h = tf.nn.elu(h)
            
            with tf.variable_scope('conv3_in_block'):
                h = tf.layers.conv2d(h, o_channel, kernel_size=[1,1], strides=1, padding='SAME')
                h = tf.layers.batch_normalization(h)

            if downsampling:
                x = tf.layers.conv2d(x, o_channel, kernel_size=[1,1], strides=stride, padding='SAME')

            do_proj = tf.shape(x)[3] != o_channel
            if do_proj:
                x = tf.layers.conv2d(x, filters=o_channel, kernel_size=[1,1], strides=1, padding='SAME', activation=None)
                return tf.nn.elu(h + x)
            else:
                return tf.nn.elu(h + x)


    def build_net(self, x_img, layer_n):
        x = x_img
        ori_shape = x_img.shape[1:-1]

        residual_block = self.residual_block if False else self.residual_block_v2

        with tf.variable_scope("conv0"): # init conv layer
            x = tf.layers.conv2d(x, filters=32, kernel_size=[3,3], strides=1, padding='SAME', activation=tf.nn.elu)
            x = tf.layers.max_pooling2d(x, pool_size=3, strides=1, padding='SAME')

        with tf.variable_scope("conv1"):
            for i in range(layer_n):
                x = residual_block(x, o_channel=256, name="resblock{}".format(i+1))
                assert (np.array(x.shape[1:-1]) == np.array(x_img.shape[1:-1])).all(), '{}, {}'.format(np.array(x.shape[1:-1]), np.array(x_img.shape[1:-1]))

        with tf.variable_scope("conv2"):
            for i in range(layer_n):
                x = residual_block(x, o_channel=512, downsampling=(i==0), name="resblock{}".format(i+1))
                assert (np.array(x.shape[1:-1]) == np.array(x_img.shape[1:-1]) // 2).all(), '{}, {}'.format(np.array(x.shape[1:-1]), np.array(x_img.shape[1:-1]))

        with tf.variable_scope("fc"):
            x = tf.reduce_mean(x, axis=[1,2]) # global average pooling

            logits = tf.sigmoid(tf.layers.dense(x, self.out_shape[0], name="logits"))

        return logits
