import tensorflow as tf
import numpy as np
import time
import os
import argparse

import DataProcess
import SegmentModel

parser = argparse.ArgumentParser(description='Skeleton-based action segment RNN model.')

parser.add_argument('-d', '--data_path', type=str, default='data',
                    help='Path to Dataset (default:\"../data\")')

parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Number of Epochs (default:10)')
parser.add_argument('-b', '--batch_size', type=int, default=15,
                    help='Batch Size (default:15)')
parser.add_argument('-rs', '--rnn_size', type=int, default=50,
                    help='RNN Size (default:50)')
parser.add_argument('-rl', '--num_layers', type=int, default=2,
                    help='Number of Layers (default:2)')
parser.add_argument('-ds', '--decoder_steps', type=int, default=3,
                    help='Steps of decoding (default:3)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='Learning Rate (default:0.001)')
parser.add_argument('-f', '--frames', type=int, default=20,
                    help='Number of frames in one sequence (default:20)')
parser.add_argument('--display_step', type=int, default=20,
                    help='Display loss for every N batches (default:20)')


parser.add_argument('-jnum', '--num_joint', type=int, default=18,
                    help='Number of joints (default:18)')
parser.add_argument('-jdim', '--coord_dim', type=int, default=2,
                    help='Dimension of joint coordinate (default:2)')

args = parser.parse_args()



if __name__ == '__main__':
    # Input size of each steps
    input_size = args.num_joint*args.coord_dim

    # Loading data
    DataLoader = DataProcess.DataProcess(path=args.data_path, 
                                         batch_size=args.batch_size, 
                                         input_size=input_size, 
                                         decoder_steps=args.decoder_steps)
    # Define graph
    train_graph = tf.Graph()
    with train_graph.as_default():
        
        # Get placeholder
        (input_data, targets, lr) = SegmentModel.get_inputs()
        
        # Build Sequence to Sequence Model 
        print('==> Creating Model...\n')
        Net = SegmentModel.Seq2Seq( input_data=input_data, targets=targets,
                                    seq_length=args.frames, rnn_size=args.rnn_size, num_layers=args.num_layers, batch_size=args.batch_size, 
                                    input_size=input_size, decoder_steps=args.decoder_steps)

        # Get Model Result
        training_decoder_output   = Net.decoder.training_decoder_output
        predicting_decoder_output = Net.decoder.predicting_decoder_output
        
        training_logits   = tf.identity(training_decoder_output,  name='train_result')   # rnn_output
        predicting_logits = tf.identity(predicting_decoder_output, name='inference_result')   # sample_id

        with tf.name_scope("optimization"):
            
            # Loss function MSE
            cost = tf.reduce_mean(tf.square(tf.subtract(training_logits, targets)))

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    
    ## Training
    print('==> Start Training...\n')
    checkpoint = "./trained_model.ckpt"

    # Create batch generator of training data
    print('Validate Generator Created')
    valid_batch_generator = DataLoader.Batch_Generator(DataLoader.valid_set['source'], DataLoader.valid_set['target'], args.frames)
    # Get validation batch
    (valid_sources_batch, valid_targets_batch, _) = next(valid_batch_generator)

    # Create Session 
    with tf.Session(graph=train_graph) as sess:
        # Init weight
        print('\n\nInit variables...')
        sess.run(tf.global_variables_initializer())
        # Training Loop
        for epoch_i in range(1, args.epochs+1):
            print('Epoch {} start...'.format(epoch_i))
            # Create batch generator of training data
            print(' - Training Generator Created')
            train_batch_generator = DataLoader.Batch_Generator( DataLoader.train_set['source'], DataLoader.train_set['target'], args.frames)
            # Get training batch
            for batch_i, (sources_batch, targets_batch, num_batch) in enumerate(train_batch_generator):
                #print('Batch {} start...'.format(batch_i))

                _, loss = sess.run( [train_op, cost],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     lr: args.learning_rate})

                if batch_i % args.display_step == 0:
                    
                    # Calculate validation loss
                    validation_loss = sess.run( [cost],
                        {input_data: valid_sources_batch,
                         targets: valid_targets_batch,
                         lr: args.learning_rate})
                    
                    print(' - Epoch {:>3}/{} | Batch {:>4}/{} | Training Loss: {:>6.3f} | Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  args.epochs, 
                                  batch_i, 
                                  num_batch, 
                                  loss, 
                                  validation_loss[0]))
    
        os._exit(0)
        
        
        # Save model
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')