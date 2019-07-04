
'''
# -------------------------------------- #
    def test_cnn(args, DataLoader)
    def cnn_graph(args, num_batch)
    def train_cnn(args, DataLoader)
# -------------------------------------- #
'''

import tensorflow as tf
import numpy as np
import argparse
import time

import Resnet
import DataProcess
import trainer
from utils import oblique_mean, draw


def test_cnn(args, DataLoader):
    print('==> Start Testing...\n')
    infer_data = DataLoader.train_set['source']
    infer_targ = DataLoader.train_set['target']
    # infer_data = DataLoader.valid_set['source']
    # infer_targ = DataLoader.valid_set['target']

    (activity, infer_data) = list(infer_data.items())[0]
    (activity, infer_targ) = list(infer_targ.items())[0]
    length = infer_data.shape[0] - (args.in_frames-1)
    print('infer_data shape:', infer_data.shape)

    # checkpoint = "./model/best_model.ckpt"
    checkpoint = "./model/trained_model.ckpt"

    loaded_graph = tf.Graph()
    answer_logits = np.zeros((length, args.out_band))   
    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('resnet/X:0')
        logits = loaded_graph.get_tensor_by_name('train_result:0')
        
        print('Start Session Run')
        start_time = time.time()
        for i in range(length):
            indata = infer_data[i:i+args.in_frames, :]
            answer_logits[i] = sess.run(logits, { input_data: np.array([indata])})[0,0] # first output, first batch
        print('time: {:6.2f}(s)'.format(time.time() - start_time))

    a_logits = oblique_mean(answer_logits)
    draw(args=args, result=a_logits, gt=infer_targ.reshape(3,-1)[-1])
    pass

def cnn_graph(args, num_total_steps):
    train_graph = tf.Graph()
    with train_graph.as_default():

        # Build Resnet Model 
        print('==> Creating Model...\n')
        Net = Resnet.ResNet(name="resnet", layer_n=5)
        input_data = Net.X
        targets = Net.y

        # Get Model Result
        Net_output = Net.logits
        training_logits = tf.identity(Net_output, name='train_result')   # cnn_output

        with tf.name_scope("optimization"):
            # Loss function
            # penalty = tf.cast(targets > 0, dtype=tf.float32) * 20 + 1
            # cost = tf.reduce_mean(tf.square(penalty * tf.subtract(training_logits, targets)))

            cost = tf.reduce_sum(tf.square(tf.subtract(training_logits, targets)))

            # Optimizer
            global_step = tf.Variable(0, trainable=False)
            boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
            values = [args.learning_rate, args.learning_rate/2, args.learning_rate/4]

            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(cost, global_step=global_step, name="train_op")
            
        pred = tf.argmax(training_logits, axis=1, name="inference_result")
        prob = tf.nn.softmax(training_logits, name="inference_prob")
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, tf.argmax(targets, axis=1)), tf.float32), name="accuracy")
        
        keep_rate = tf.Variable(0, trainable=False) # useless

        useful = { 'train_op':train_op, 
                   'cost':cost, 
                   'input_data':input_data,
                   'targets':targets,
                   'keep_rate':keep_rate,
                   'optimizer':optimizer}
    return train_graph, useful

def train_cnn(args, DataLoader):
    num_batch = DataLoader.Get_num_batch(DataLoader.train_set['source'], args.in_frames)

    # Define graph
    num_total_steps = args.epochs * num_batch
    (train_graph, useful) = cnn_graph(args, num_total_steps)

    ## Training
    trainer.train_loop(args, train_graph, DataLoader, num_batch, useful)
    pass
