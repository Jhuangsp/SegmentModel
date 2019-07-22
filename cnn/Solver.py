## CNN Solver


import tensorflow as tf
import numpy as np
import argparse
import time
import os 

import DataProcess
import cnn.Resnet
from utils.dynamic_time_warpping.dtw import dtw
from scipy.signal import find_peaks
from utils.utils import oblique_mean, draw

class Solver(object):
    """docstring for Solver"""
    def __init__(self):
        super(Solver, self).__init__()

    def train(self, args, DataLoader, net):
        print('==> Start Training...\n')
        checkpoint = "./model/trained_model.ckpt"
        best_point = "./model/best_model.ckpt"

        # Create Session 
        with tf.Session(graph=net.train_graph) as sess:
            saver = tf.train.Saver()
            best = 1e+6

            train_op   = net.train_op
            cost       = net.cost
            input_data = net.X
            targets    = net.y
            training_logits = net.training_logits
            keep_rate  = net.keep_rate
            optimizer  = net.optimizer

            # Init weight
            sess.run(tf.global_variables_initializer())

            # Training Loop
            for epoch_i in range(1, args.epochs+1):
                print('Epoch {} start...'.format(epoch_i))
                # Create batch generator of training data
                print(' - Training Generator Created')
                if args.model=='cnn':
                    train_batch_generator = DataLoader.Batch_Generator_Resnet( DataLoader.train_set['source'], DataLoader.train_set['target'], args.in_frames, args.out_band)
                else:
                    train_batch_generator = DataLoader.Batch_Generator( DataLoader.train_set['source'], DataLoader.train_set['target'], args.in_frames, args.out_band)
                # Get training batch
                for batch_i, (sources_batch, targets_batch, num_batch) in enumerate(train_batch_generator):

                    _, loss = sess.run( [train_op, cost],
                        {input_data: sources_batch,
                         targets: targets_batch,
                         keep_rate: 0.9})

                    if batch_i % args.display_step == 0:
                        
                        if args.model=='cnn':
                            valid_batch_generator = DataLoader.Batch_Generator_Resnet(DataLoader.valid_set['source'], DataLoader.valid_set['target'], args.in_frames, args.out_band, training=False)
                        else:
                            valid_batch_generator = DataLoader.Batch_Generator(DataLoader.valid_set['source'], DataLoader.valid_set['target'], args.in_frames, args.out_band, training=False)
                        
                        ''' Validatio loss
                        all_loss = []
                        dtws = []
                        for (valid_sources_batch, valid_targets_batch, _) in valid_batch_generator:

                            # Calculate validation loss
                            validation_loss, LR, r1, r2 = sess.run( [cost, optimizer._lr, targets, training_logits],
                                {input_data: valid_sources_batch,
                                 targets: valid_targets_batch,
                                 keep_rate: 0.9})
                            all_loss.append(validation_loss)

                        print(' - Epoch {:>3}/{} | Batch {:>4}/{} | LR: {:>4.3e} | Train Loss: {:>6.3f} | Valid loss: {:>6.3f}'
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
                            best_e = epoch_i
                            print('Best Model at epoch {}, Valid loss: {:>6.3f}'.format(best_e, best))
                        '''
                        
                        # ''' Peak detect + DTW
                        infer_data = DataLoader.valid_set['source']
                        infer_targ = DataLoader.valid_set['target']

                        (activity, infer_data) = list(infer_data.items())[0]
                        (activity, infer_targ) = list(infer_targ.items())[0]
                        length = infer_data.shape[0] - (args.in_frames-1)

                        answer_logits = np.zeros((length, args.out_band))
                        
                        for i in range(length):
                            indata = infer_data[i:i+args.in_frames, :]
                            LR, validation_result= sess.run([optimizer._lr, training_logits], { input_data: np.array([indata])})
                            answer_logits[i] = validation_result[0,0]

                        euclidean_norm = lambda x, y: np.abs(x - y)
                        a_logits = oblique_mean(answer_logits)
                        padsize = (args.in_frames - args.out_band) // 2
                        a_logits = np.pad(a_logits, (padsize, padsize), 'edge')
                        # peak
                        rtpeaks, _ = find_peaks(a_logits, height=0)
                        gtpeaks, _ = find_peaks(infer_targ.reshape(3,-1)[-1], height=0)
                        if rtpeaks.size == 0 or gtpeaks.size == 0:
                            rtpeaks = np.insert(rtpeaks, 0, 0)
                            gtpeaks = np.insert(gtpeaks, 0, 0)
                        d, cost_matrix, acc_cost_matrix, path = dtw(gtpeaks, rtpeaks, dist=euclidean_norm)
                        
                        print(' - Epoch {:>3}/{} | Batch {:>4}/{} | LR: {:>4.3e} | Train Loss: {:>6.3f} | Valid DTW: {:>6.3f}'
                              .format(epoch_i,
                                      args.epochs, 
                                      batch_i, 
                                      num_batch,
                                      LR, 
                                      loss,
                                      d))
                        
                        if d < best:
                            saver.save(sess, best_point)
                            best = d
                            best_e = epoch_i
                            print('Best Model at epoch {}, dtw {:>6.3f}'.format(best_e, best))
                        # '''
            print('Best Model at epoch {}, Valid loss: {:>6.3f}'.format(best_e, best))
            # Save model
            saver.save(sess, checkpoint)
            print('Model Trained and Saved\n')
        pass

    def test(self, args, DataLoader):
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

            input_data = loaded_graph.get_tensor_by_name('resnet/X:0')
            logits = loaded_graph.get_tensor_by_name('train_result:0')
            
            print('Start Session Run')
            start_time = time.time()
            for i in range(length):
                indata = infer_data[i:i+args.in_frames, :]
                answer_logits[i] = sess.run(logits, { input_data: np.array([indata])})[0,0] # first output, first batch
            print('time: {:6.2f}(s)'.format(time.time() - start_time))

        a_logits = oblique_mean(answer_logits)
        padsize = (args.in_frames - args.out_band) // 2
        a_logits = np.pad(a_logits, (padsize, padsize), 'edge')
        draw(args=args, result=a_logits, gt=infer_targ.reshape(3,-1)[-1])