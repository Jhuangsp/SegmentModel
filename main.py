import tensorflow as tf
import numpy as np
import time
import os,sys
import argparse
import datetime
from matplotlib import pyplot as plt

import DataProcess
import SegmentModel
from utils import oblique_mean

parser = argparse.ArgumentParser(description='Skeleton-based action segment RNN model.', fromfile_prefix_chars='@')

# Data parameter
parser.add_argument('-d', '--data_path', type=str, default='data',
                    help='Path to Dataset (default:\"../data\")')
parser.add_argument('-jnum', '--num_joint', type=int, default=18,
                    help='Number of joints (default:18)')
parser.add_argument('-jdim', '--coord_dim', type=int, default=2,
                    help='Dimension of joint coordinate (default:2)')

# Training parameter
parser.add_argument('-e', '--epochs', type=int, default=100,
                    help='Number of Epochs (default:100)')
parser.add_argument('-b', '--batch_size', type=int, default=15,
                    help='Batch Size (default:15)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                    help='Learning Rate (default:0.0001)')

# Display parameter
parser.add_argument('--display_step', type=int, default=50,
                    help='Display loss for every N batches (default:50)')
parser.add_argument('--test', action="store_true",
                    help='Only do inference')
parser.add_argument('--info', type=str,
                    help='Information about this training.')

# Model structure parameter
parser.add_argument('-rs', '--rnn_size', type=int, default=50,
                    help='RNN Size (default:50)')
parser.add_argument('-rl', '--num_layers', type=int, default=4,
                    help='Number of Layers (default:4)')
parser.add_argument('-ds', '--decoder_steps', type=int, default=3,
                    help='Steps of decoding (default:3)')
parser.add_argument('-if', '--in_frames', type=int, default=20,
                    help='Number of frames in one input sequence (default:20)')
parser.add_argument('-ob', '--out_band', type=int, default=10,
                    help='Number of output-band frames in the mid of input sequence (default:10)')

args = parser.parse_args()

if args.in_frames <= args.out_band:
    parser.error('-if must larger than -ob.')
elif (args.in_frames-args.out_band)%2 != 0:
    parser.error('The value -if larger than -ob must divisible by 2.')


if args.info != None:
    # save args
    with open('./model/command_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))
    # save information
    now = datetime.datetime.now()
    current_time = '{:04d}_{:02d}_{:02d}_{:02d}{:02d}{:02d}\n'.format(
        now.year, 
        now.month, 
        now.day, 
        now.hour, 
        now.minute, 
        now.second)
    with open('./model/info.txt', 'w') as out_file:
        out_file.write(current_time)
        out_file.write(args.info)


if __name__ == '__main__':
    # Input size of each steps
    input_size = args.num_joint*args.coord_dim

    # Loading data
    DataLoader = DataProcess.DataProcess(path=args.data_path, 
                                         batch_size=args.batch_size, 
                                         input_size=input_size, 
                                         decoder_steps=args.decoder_steps)
    if not args.test:
        train_batch_generator = DataLoader.Batch_Generator( DataLoader.train_set['source'], DataLoader.train_set['target'], args.in_frames, args.out_band)
        (_, _, num_batch) = next(train_batch_generator)

        # Define graph
        train_graph = tf.Graph()
        with train_graph.as_default():
            
            # Get placeholder
            (input_data, targets, keep_rate) = SegmentModel.get_inputs()
            
            # Build Sequence to Sequence Model 
            print('==> Creating Model...\n')
            Net = SegmentModel.Seq2Seq( input_data=input_data, targets=targets, keep_rate=keep_rate,
                                        seq_length=args.in_frames, out_length=args.out_band, rnn_size=args.rnn_size, num_layers=args.num_layers, batch_size=args.batch_size, 
                                        input_size=input_size, decoder_steps=args.decoder_steps)

            # Get Model Result
            training_decoder_output   = Net.decoder.training_decoder_output
            predicting_decoder_output = Net.decoder.predicting_decoder_output
            
            training_logits   = tf.identity(training_decoder_output,  name='train_result')   # rnn_output
            predicting_logits = tf.identity(predicting_decoder_output, name='inference_result')   # sample_id

            with tf.name_scope("optimization"):
                
                # Loss function MSE/SSE

                penalty = tf.cast(targets > 0, dtype=tf.float32) * 20 + 1
                cost = tf.reduce_mean(tf.square(penalty * tf.subtract(training_logits, targets)))

                # Optimizer
                global_step = tf.Variable(0, trainable=False)

                num_total_steps = args.epochs*num_batch
                boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
                values = [args.learning_rate, args.learning_rate/2, args.learning_rate/4]

                learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                # optimizer = tf.train.AdamOptimizer(lr)
                add_global = global_step.assign_add(1)

                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)
        
        ## Training
        print('==> Start Training...\n')
        checkpoint = "./model/trained_model.ckpt"
        best_point = "./model/best_model.ckpt"

        # # Create batch generator of training data
        # print('Validate Generator Created')
        # valid_batch_generator = DataLoader.Batch_Generator(DataLoader.valid_set['source'], DataLoader.valid_set['target'], args.in_frames, args.out_band, training=False)
        # # Get validation batch
        # (valid_sources_batch, valid_targets_batch, _) = next(valid_batch_generator)

        # Create Session 
        with tf.Session(graph=train_graph) as sess:
            saver = tf.train.Saver()
            best = 10.

            # Init weight
            print('\n\nInit variables...')
            sess.run(tf.global_variables_initializer())
            # Training Loop
            for epoch_i in range(1, args.epochs+1):
                print('Epoch {} start...'.format(epoch_i))
                # Create batch generator of training data
                print(' - Training Generator Created')
                train_batch_generator = DataLoader.Batch_Generator( DataLoader.train_set['source'], DataLoader.train_set['target'], args.in_frames, args.out_band)
                # Get training batch
                for batch_i, (sources_batch, targets_batch, num_batch) in enumerate(train_batch_generator):
                    #print('Batch {} start...'.format(batch_i))

                    _, loss, _ = sess.run( [train_op, cost, add_global],
                        {input_data: sources_batch,
                         targets: targets_batch,
                         keep_rate: 0.9})

                    if batch_i % args.display_step == 0:
                        
                        valid_batch_generator = DataLoader.Batch_Generator(DataLoader.valid_set['source'], DataLoader.valid_set['target'], args.in_frames, args.out_band, training=False)
                        all_loss = []
                        for (valid_sources_batch, valid_targets_batch, _) in valid_batch_generator:

                            # Calculate validation loss
                            validation_loss, LR = sess.run( [cost, optimizer._lr],
                                {input_data: valid_sources_batch,
                                 targets: valid_targets_batch,
                                 keep_rate: 0.9})
                            all_loss.append(validation_loss)
                        
                        print(' - Epoch {:>3}/{} | Batch {:>4}/{} | LR: {:>4.3e} | Training Loss: {:>6.3f} | Validation loss: {:>6.3f}'
                              .format(epoch_i,
                                      args.epochs, 
                                      batch_i, 
                                      num_batch, 
                                      LR, 
                                      loss, 
                                      sum(all_loss)/len(all_loss)))
                        if sum(all_loss)/len(all_loss) < best:
                            saver.save(sess, best_point)
                            best = sum(all_loss)/len(all_loss)
                            print('Best Model at epoch {}, Loss {:>6.3f}'.format(epoch_i, best))
            
            # Save model
            saver.save(sess, checkpoint)
            print('Model Trained and Saved\n')



    # Testing
    print('==> Start Testing...\n')
    # infer_data = DataLoader.train_set['source']
    # infer_targ = DataLoader.train_set['target']
    infer_data = DataLoader.valid_set['source']
    infer_targ = DataLoader.valid_set['target']

    (activity, infer_data) = list(infer_data.items())[0]
    (activity, infer_targ) = list(infer_targ.items())[0]
    length = infer_data.shape[0] - (args.in_frames-1)
    print('infer_data shape:', infer_data.shape)

    # checkpoint = "./model/best_model.ckpt"
    checkpoint = "./model/trained_model.ckpt"
    result = "./model/result.npy"
    gt = "./model/gt.npy"

    loaded_graph = tf.Graph()
    answer_logits = np.zeros((length, args.out_band))   
    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('inputs:0')
        keep_rate = loaded_graph.get_tensor_by_name('keep_rate:0')
        logits = loaded_graph.get_tensor_by_name('inference_result:0')
        
        print('Start')
        for i in range(length):
            indata = infer_data[i:i+args.in_frames, :]
            answer_logits[i] = sess.run(logits, { input_data: np.tile(indata, (15,1,1)), keep_rate: 1.0 })[0,0]
            # answer_logits[i] = sess.run(logits, { input_data: np.tile(indata, (30,1,1)), keep_rate: 1.0 })[0,0]

    a_logits = oblique_mean(answer_logits)
    ls = np.arange(a_logits.shape[0])
    plt.scatter(ls, a_logits, color='orange')
    plt.plot(ls, a_logits, color='orange')
    np.save(result, a_logits)

    ls = np.arange(len(infer_targ.reshape(3,-1)[-1]))
    plt.scatter(ls, infer_targ.reshape(3,-1)[-1])
    plt.plot(ls, infer_targ.reshape(3,-1)[-1])
    np.save(gt, infer_targ.reshape(3,-1)[-1])

    plt.xlabel('Frame')
    plt.ylabel('Probability of Changing Point Frame')
    # plt.xlim([30, 70])
    plt.ylim([-0.1, 1.1])
    plt.show()