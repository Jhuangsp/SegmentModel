
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

import DataProcess

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
        # add_global = useful['add_global']
        input_data = useful['input_data']
        targets    = useful['targets']
        keep_rate  = useful['keep_rate']
        optimizer  = useful['optimizer']

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

                # _, loss, _ = sess.run( [train_op, cost, add_global],
                _, loss = sess.run( [train_op, cost],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     keep_rate: 0.9})

                if batch_i % args.display_step == 0:
                    
                    if args.model=='cnn':
                        valid_batch_generator = DataLoader.Batch_Generator_Resnet(DataLoader.valid_set['source'], DataLoader.valid_set['target'], args.in_frames, args.out_band, training=False)
                    else:
                        valid_batch_generator = DataLoader.Batch_Generator(DataLoader.valid_set['source'], DataLoader.valid_set['target'], args.in_frames, args.out_band, training=False)
                    all_loss = []
                    for (valid_sources_batch, valid_targets_batch, _) in valid_batch_generator:

                        # Calculate validation loss
                        validation_loss, LR = sess.run( [cost, optimizer._lr],
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
                        print('Best Model at epoch {}, Loss {:>6.3f}'.format(epoch_i, best))
        
        # Save model
        saver.save(sess, checkpoint)
        print('Model Trained and Saved\n')
    pass

    