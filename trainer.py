
'''
# -------------------------------------- #
    All model share same fuction
    def train_loop()
    def train(args, DataLoader)
    def test(args, DataLoader)
# -------------------------------------- #
'''
import tensorflow as tf
import numpy as np
import argparse
import time
import os

import DataProcess
from utils.dynamic_time_warpping.dtw import dtw
from scipy.signal import find_peaks
from utils.utils import oblique_mean

def train_loop(args, train_graph, DataLoader, num_batch, useful):
    print('==> Start Training...\n')
    checkpoint = "./model/trained_model.ckpt"
    best_point = "./model/best_model.ckpt"

    # Create Session 
    with tf.Session(graph=train_graph) as sess:
        saver = tf.train.Saver()
        best = 1e+6

        train_op   = useful['train_op']
        cost       = useful['cost']
        input_data = useful['input_data']
        targets    = useful['targets']
        training_logits = useful['training_logits'],
        keep_rate  = useful['keep_rate']
        optimizer  = useful['optimizer']
        show = useful['show']

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
                    '''
                    all_loss = []
                    dtws = []
                    for (valid_sources_batch, valid_targets_batch, _) in valid_batch_generator:

                        # Calculate validation loss
                        validation_loss, LR, r1, r2 = sess.run( [cost, optimizer._lr, targets, training_logits],
                            {input_data: valid_sources_batch,
                             targets: valid_targets_batch,
                             keep_rate: 0.9})
                        all_loss.append(validation_loss)

                        # peak detect
                        euclidean_norm = lambda x, y: np.abs(x - y)
                        for i in range(args.batch_size):
                            gtpeaks, _ = find_peaks(r1[i], height=0)
                            rtpeaks, _ = find_peaks(r2[0][i], height=0)
                            if rtpeaks.size == 0 or gtpeaks.size == 0:
                                rtpeaks = np.insert(rtpeaks, 0, 0)
                                gtpeaks = np.insert(gtpeaks, 0, 0)

                            # DTW
                            d, cost_matrix, acc_cost_matrix, path = dtw(gtpeaks, rtpeaks, dist=euclidean_norm)
                            # d, cost_matrix, acc_cost_matrix, path = dtw(r1[i], r2[0][i], dist=euclidean_norm)
                            dtws.append(d)
                    '''
                    # ------------------- #
                    infer_data = DataLoader.valid_set['source']
                    infer_targ = DataLoader.valid_set['target']

                    (activity, infer_data) = list(infer_data.items())[0]
                    (activity, infer_targ) = list(infer_targ.items())[0]
                    length = infer_data.shape[0] - (args.in_frames-1)

                    answer_logits = np.zeros((length, args.out_band))
                    
                    for i in range(length):
                        indata = infer_data[i:i+args.in_frames, :]
                        answer_logits[i] = sess.run(training_logits, { input_data: np.array([indata])})[0][0,0] # first output, first batch

                    euclidean_norm = lambda x, y: np.abs(x - y)
                    a_logits = oblique_mean(answer_logits)
                    gtpeaks, _ = find_peaks(a_logits, height=0)
                    rtpeaks, _ = find_peaks(infer_targ.reshape(3,-1)[-1], height=0)
                    if rtpeaks.size == 0 or gtpeaks.size == 0:
                        rtpeaks = np.insert(rtpeaks, 0, 0)
                        gtpeaks = np.insert(gtpeaks, 0, 0)
                    d, cost_matrix, acc_cost_matrix, path = dtw(gtpeaks, rtpeaks, dist=euclidean_norm)
                    
                    # ------------------- #
                    print(' - Epoch {:>3}/{} | Batch {:>4}/{} | LR: {:>4.3e} | Train Loss: {:>6.3f} | Valid loss: {:>6.3f}, dtw: {:>6.3f}'
                          .format(epoch_i,
                                  args.epochs, 
                                  batch_i, 
                                  num_batch, 0,
                                  # LR, 
                                  loss, 0,
                                  # sum(all_loss)/len(all_loss),
                                  d))
                                  # sum(dtws)/len(dtws)))
                    # if sum(all_loss)/len(all_loss) < best:
                    #     # saver.save(sess, best_point)
                    #     best = sum(all_loss)/len(all_loss)
                    #     print('Best Model at epoch {}, Loss {:>6.3f}'.format(epoch_i, best))
                    # if sum(dtws)/len(dtws) < best:
                    #     saver.save(sess, best_point)
                    #     best = sum(dtws)/len(dtws)
                    #     print('Best Model at epoch {}, dtw {:>6.3f}'.format(epoch_i, best))
                    if d < best:
                        saver.save(sess, best_point)
                        best = d
                        print('Best Model at epoch {}, dtw {:>6.3f}'.format(epoch_i, best))
        
        # Save model
        saver.save(sess, checkpoint)
        print('Model Trained and Saved\n')
    pass

    