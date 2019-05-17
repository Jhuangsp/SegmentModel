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

parser.add_argument('-e', '--epochs', type=int, default=60,
                    help='Number of Epochs (default:60)')
parser.add_argument('-b', '--batch_size', type=int, default=3,
                    help='Batch Size (default:8)')
parser.add_argument('-rs', '--rnn_size', type=int, default=50,
                    help='RNN Size (default:50)')
parser.add_argument('-rl', '--num_layers', type=int, default=2,
                    help='Number of Layers (default:2)')
parser.add_argument('-ds', '--decoder_steps', type=int, default=3,
                    help='Steps of decoding (default:3)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='Learning Rate (default:0.001)')
parser.add_argument('-jnum', '--num_joint', type=int, default=18,
                    help='Number of joints (default:18)')
parser.add_argument('-jdim', '--coord_dim', type=int, default=3,
                    help='Dimension of joint coordinate (default:3)')

args = parser.parse_args()

# Input size of each steps
input_size = args.num_joint*args.coord_dim


if __name__ == '__main__':

    # Loading data
    DataLoader = DataProcess.DataProcess(path=args.data_path, 
                                         batch_size=args.batch_size, 
                                         input_size=input_size, 
                                         decoder_steps=args.decoder_steps)
    os._exit(0)
    # Define graph
    print('==> Defining Graph...\n')
    train_graph = tf.Graph()
    with train_graph.as_default():
        
        # Get placeholder
        ( input_data,
          targets,
          lr,
          source_seq_len,
          target_seq_len,
          max_target_seq_len ) = Seq2SeqModel.get_inputs()
        
        # 建立 Sequence to Sequence Model 
        print('==> Creating Model...\n')
        Net = Seq2SeqModel.Seq2Seq(input_data, targets, lr, 
                                    source_seq_len, 
                                    target_seq_len, 
                                    max_target_seq_len,
                                    DataLoader,
                                    args.rnn_size, args.num_layers, args.batch_size, input_size)
        # 取出 Model 結果
        training_decoder_output   = Net.decoder.training_decoder_output
        predicting_decoder_output = Net.decoder.predicting_decoder_output
        
        training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')   # rnn_output
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')   # sample_id
        
        masks = tf.sequence_mask(target_seq_len, max_target_seq_len, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

    
    ## Training
    print('==> Start Training...\n')
    display_step = 2 # display loss for every 2 batches
    checkpoint = "./trained_model.ckpt"

    # Create batch generator of training data
    valid_batch_generator = DataLoader.Batch_Generator(
                                    DataLoader.valid_set['source'], 
                                    DataLoader.valid_set['target'])
    # Get validation batch
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(valid_batch_generator)

    # 建立 Session 
    with tf.Session(graph=train_graph) as sess:
        # 初始化權重
        sess.run(tf.global_variables_initializer())
        # Training Loop
        for epoch_i in range(1, args.epochs+1):
            # Create batch generator of training data
            train_batch_generator = DataLoader.Batch_Generator(
                                            DataLoader.train_set['source'], 
                                            DataLoader.train_set['target'])
            # Get training batch
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(train_batch_generator):
                # targets_batch: type=list, shape=[args.batch_size = 128, steps = 7]
                # sources_batch: type=list, shape=[args.batch_size = 128, steps = 7]
                # targets_lengths: type=list, shape=[args.batch_size = 128]
                # sources_lengths: type=list, shape=[args.batch_size = 128]

                _, loss = sess.run( [train_op, cost],
                    {  input_data: sources_batch,
                       targets: targets_batch,
                       lr: args.learning_rate,
                       target_seq_len: targets_lengths,
                       source_seq_len: sources_lengths  })

                if batch_i % display_step == 0:
                    
                    # 计算validation loss
                    validation_loss = sess.run( [cost],
                      {  input_data: valid_sources_batch,
                         targets: valid_targets_batch,
                         lr: args.learning_rate,
                         target_seq_len: valid_targets_lengths,
                         source_seq_len: valid_sources_lengths  })
                    
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  args.epochs, 
                                  batch_i, 
                                  len(DataLoader.train_set['source']) // args.batch_size, 
                                  loss, 
                                  validation_loss[0]))

        
        
        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')

    # # Testing
    # # 輸入一個單字
    # input_word = 'common'
    # text = source_to_seq(input_word, DataLoader.source_dict['letter2int'])

    # checkpoint = "./trained_model.ckpt"

    # loaded_graph = tf.Graph()
    # with tf.Session(graph=loaded_graph) as sess:
    #     # Load model
    #     loader = tf.train.import_meta_graph(checkpoint + '.meta')
    #     loader.restore(sess, checkpoint)

    #     input_data = loaded_graph.get_tensor_by_name('inputs:0')
    #     logits = loaded_graph.get_tensor_by_name('predictions:0')
    #     source_seq_len = loaded_graph.get_tensor_by_name('source_seq_len:0')
    #     target_seq_len = loaded_graph.get_tensor_by_name('target_seq_len:0')
        
    #     answer_logits = sess.run(logits, {input_data: [text]*args.batch_size, 
    #                                       target_seq_len: [len(input_word)]*args.batch_size, 
    #                                       source_seq_len: [len(input_word)]*args.batch_size})[0] 


    # pad = DataLoader.source_dict['letter2int']["<PAD>"] 

    # print('原始輸入:', input_word)

    # print('\nSource')
    # print('  Word 編號:    {}'.format([i for i in text]))
    # print('  Input Words: {}'.format(' '.join([DataLoader.source_dict['int2letter'][i] for i in text])))

    # print('\nTarget')
    # print('  Word 編號:       {}'.format([i for i in answer_logits if i != pad]))
    # print('  Response Words: {}'.format(" ".join([DataLoader.target_dict['int2letter'][i] for i in answer_logits if i != pad])))