
'''
# -------------------------------------- #
    def test_rnn(args, DataLoader)
    def rnn_graph(args, num_batch)
    def train_rnn(args, DataLoader)
# -------------------------------------- #
'''

import tensorflow as tf
import numpy as np
import argparse
import time

import SegmentModel
import DataProcess
import trainer
from utils.utils import oblique_mean, draw

def test_rnn(args, DataLoader):
    print('==> Start Testing...\n')
    # infer_data = DataLoader.train_set['source']
    # infer_targ = DataLoader.train_set['target']
    infer_data = DataLoader.valid_set['source']
    infer_targ = DataLoader.valid_set['target']

    (activity, infer_data) = list(infer_data.items())[0]
    (activity, infer_targ) = list(infer_targ.items())[0]
    length = infer_data.shape[0] - (args.in_frames-1)
    print('infer_data shape:', infer_data.shape)

    checkpoint = "./model/best_model.ckpt"
    # checkpoint = "./model/trained_model.ckpt"

    loaded_graph = tf.Graph()
    answer_logits = np.zeros((length, args.out_band))   
    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('inputs:0')
        keep_rate = loaded_graph.get_tensor_by_name('keep_rate:0')
        logits = loaded_graph.get_tensor_by_name('inference_result:0')
        
        print('Start Session Run')
        start_time = time.time()
        for i in range(length):
            indata = infer_data[i:i+args.in_frames, :]
            answer_logits[i] = sess.run(logits, { input_data: np.tile(indata, (1,1,1)), keep_rate: 1.0 })[0,0] # first output, first batch
        print('time: {:6.2f}(s)'.format(time.time() - start_time))

    a_logits = oblique_mean(answer_logits)
    draw(args=args, result=a_logits, gt=infer_targ.reshape(3,-1)[-1])
    pass

def rnn_graph(args, num_total_steps):
    input_size = args.num_joint*args.coord_dim
    train_graph = tf.Graph()
    with train_graph.as_default():
        
        # Get placeholder
        (input_data, targets, keep_rate) = SegmentModel.get_inputs()
        
        # Build Sequence to Sequence Model 
        Net = SegmentModel.Seq2Seq( input_data=input_data, targets=targets, keep_rate=keep_rate,
                                    seq_length=args.in_frames, out_length=args.out_band, rnn_size=args.rnn_size, num_layers=args.num_layers, batch_size=args.batch_size, 
                                    input_size=input_size, decoder_steps=args.decoder_steps)

        # Get Model Result
        training_decoder_output   = Net.decoder.training_decoder_output
        predicting_decoder_output = Net.decoder.predicting_decoder_output
        
        training_logits   = tf.identity(training_decoder_output,  name='train_result')   # rnn_output
        predicting_logits = tf.identity(predicting_decoder_output, name='inference_result')   # sample_id

        with tf.name_scope("optimization"):
            
            # Loss function MSE
            penalty = tf.cast(targets > 0, dtype=tf.float32) * 20 + 1
            cost = tf.reduce_mean(tf.square(penalty * tf.subtract(training_logits, targets)))

            # Optimizer
            global_step = tf.Variable(0, trainable=False)
            boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
            values = [args.learning_rate, args.learning_rate/2, args.learning_rate/4]

            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # add_global = global_step.assign_add(1)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step, name="train_op")

        useful = { 'train_op':train_op, 
                   'cost':cost, 
                   # 'add_global':add_global, 
                   'input_data':input_data,
                   'targets':targets,
                   'keep_rate':keep_rate,
                   'optimizer':optimizer}

    return train_graph, useful

def train_rnn(args, DataLoader):
    num_batch = DataLoader.Get_num_batch(DataLoader.train_set['source'], args.in_frames)

    # Define graph
    num_total_steps = args.epochs*num_batch
    (train_graph, useful) = rnn_graph(args, num_total_steps)

    ## Training
    trainer.train_loop(args, train_graph, DataLoader, num_batch, useful)
    pass
