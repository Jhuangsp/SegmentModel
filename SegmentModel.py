import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import os

import DataProcess


def get_inputs():
    '''
    Define all input tf.placeholder tensor
    '''
    # Data
    inputs  = tf.placeholder(tf.float32, [None, None, None], name='inputs') # (batch_size, steps, input_size)
    targets = tf.placeholder(tf.float32, [None, None], name='targets') # (batch_size, steps, 0 or 1)
    
    # Learning rate
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    # 單筆 source and target seqence's original length
    source_seq_len = tf.placeholder(tf.int32, (None,), name='source_seq_len')
    target_seq_len = tf.placeholder(tf.int32, (None,), name='target_seq_len')

    # longest target sequence length (scale)
    max_target_seq_len = tf.reduce_max(target_seq_len, name='longest_target_seq_len')
    
    return inputs, targets, learning_rate, source_seq_len, target_seq_len, max_target_seq_len

class Seq2Seq(object):
    '''
    Sequence to Sequence Model
    
    Argument：
    - input_data:                  (tf.placeholder)  padding 完的1個 batch 的輸入資料 (tensor:[batch_size, steps])
    - targets:                     (tf.placeholder)  padding 完的1個 batch 的目標資料 (tensor:[batch_size, steps])
    - lr:                          (tf.placeholder)  learning rate (tensor:scale)
    - source_seq_len:              (tf.placeholder)  input 數據序列的原始長度 (tensor:[batch_size])
    - target_seq_len:              (tf.placeholder)  target 數據序列的原始長度 (tensor:[batch_size])
    - max_target_seq_len:          (tf.placeholder)  target 數據序列的最大原始長度 (tensor:scale)
    
    - rnn_size:                                 RNN hiden state 的維度 (cells數)
    - num_layers:                               RNN的層數
    - batch_size:                               batch_size
    - input_size:                               Input size of each steps
    '''
    def __init__(self, 
                  input_data, targets, lr, source_seq_len, target_seq_len, max_target_seq_len,
                  dataloader,
                  rnn_size, num_layers, batch_size, input_size):
        super(Seq2Seq, self).__init__()

        self.input_data = input_data
        self.targets = targets
        self.lr = lr
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.max_target_seq_len = max_target_seq_len

        self.dataloader = dataloader

        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size

        # Preprocess the target data used for training
        def process_decoder_input(data, batch_size, step_input_size):
            '''
            Add START_TOKEN at the front and remove the LAST_TOKEN
                e.g. Ground Truth=[4,5,6], then Input=[start_tokens,4,5]

            Parameter:
            - data: shape=(batch_size, sequence_len)
            '''
            # START_TOKEN = np.full((batch_size, 1), 0.5)
            data_wo_ending = tf.strided_slice(data, begin=[0, 0], end=[batch_size, -1], strides=[1, 1])
            decoder_input = tf.concat([tf.fill([batch_size, 1], 0.5), data_wo_ending], axis=1)

            return decoder_input


        self.encoder = self.Encoder(self.input_data, 
                                    self.source_seq_len,
                                    self.rnn_size, 
                                    self.num_layers,
                                    self.input_size)

        # Preprocess the target data
        self.decoder_input = process_decoder_input(self.targets,  
                                            self.batch_size, 
                                            self.input_size)

        self.decoder = self.Decoder(self.encoder.encoder_state,
                                    self.decoder_input,
                                    self.target_seq_len,
                                    self.max_target_seq_len,
                                    self.dataloader.target_dict['letter2int'], 
                                    self.rnn_size,
                                    self.num_layers, 
                                    self.batch_size)

    ## Inner class
    class Encoder(object):
        '''
        Parameter:
        - input_data:               Input tensor
        - seq_length:               Steps of one input sequence (dtype=int)

        - rnn_size:                 Size of lstm hiden state
        - num_layers:               Number of multy-layer-lstm layers
        - input_size:               Input size of each steps
        
        Result:
        - encoder_output:           Short-term momory output by ALL steps [batch_size, steps, RNN_size]
        - encoder_state:            Long-term memory and Short-term momory at the end of steps 
                                        ([batch_size, RNN_size], [batch_size, RNN_size])
        '''
        def __init__(self, input_data, seq_length, 
                       rnn_size, num_layers, input_size):
            super(Seq2Seq.Encoder, self).__init__()

            self.input_data = input_data
            self.seq_length = seq_length

            self.rnn_size = rnn_size
            self.num_layers = num_layers
            self.input_size = input_size

            # init One Dense Layer
            self.encoder_dense = self.dense_layer(in_size=self.input_size, out_size=self.rnn_size) # Before Eecoder Input.

            # Result
            self.encoder_output, self.encoder_state = self.forward()

        def dense_layer(self, in_size, out_size):
            weight = tf.Variable(tf.random_normal([in_size, out_size])) # shape (in_size, out_size)
            biases = tf.Variable(tf.constant(0.1, shape=[out_size, ])) # shape (out_size, )
            return {'weight':weight, 'biases':biases, }

        def forward(self):
            # LSTM cell unit
            def get_lstm_cell(rnn_size):
                lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return lstm_cell
                        
            # Embedding
            X = tf.reshape(self.input_data, [-1, self.input_size]) # [batch_size, steps, input_size] -> [batch_size*steps, input_size]
            X = tf.matmul(X, self.encoder_dense['weight']) + self.encoder_dense['biases']
            self.encoder_embed_input = tf.reshape(X, [-1, self.seq_length, self.rnn_size])


            # Multi-layer LSTM
            self.cells = tf.contrib.rnn.MultiRNNCell([ get_lstm_cell(self.rnn_size) for _ in range(self.num_layers) ])
            
            sequence_length = tf.fill([self.batch_size], self.seq_length)
            encoder_output, encoder_state = tf.nn.dynamic_rnn(self.cells, self.encoder_embed_input, 
                                                              sequence_length=sequence_length, dtype=tf.float32)
            
            return encoder_output, encoder_state 

    class Decoder(object):
        '''
        Decoder Layer
            1. Initial state with Encoder state.
            2. Two Dense Layer.
                - One before Decoder Input. (output_size -> embedding_size)
                - Another one after the Decoder Output. (embedding_size -> output_size)
            3. Multi-Layer LSTM
            4-1. Training decoder.
                - Feed the Ground Truth as input to each steps.
                - The input are start with start_tokens, end with second last step of Ground Truth
                    - e.g. Ground Truth=[4,5,6], then Input=[start_tokens,4,5]
            4-2. Inference decoder.
                - Reuse 3-1's model parameter
                - Feed the previous step's output as now step's input.
                - Set 

        Parameter：
        - encoder_state:    encoder端輸出的狀態向量 (tensor)
        - decoder_input:    decoder端输入 (tensor) (Only used in training)

        - seq_length:       Steps of one input sequence (dtype=int)
        - decoder_steps:    Number of steps should decoder do.
        - rnn_size:         Size of lstm hiden state
        - num_layers:       Number of multy-layer-lstm layers
        - batch_size:       Batch size
        '''

        def __init__(self, encoder_state, decoder_input, 
                       seq_length, decoder_steps, rnn_size, num_layers, batch_size):
            super(Seq2Seq.Decoder, self).__init__()

            self.encoder_state = encoder_state
            self.decoder_input = decoder_input

            self.seq_length = seq_length
            self.decoder_steps = decoder_steps
            self.num_layers = num_layers
            self.rnn_size = rnn_size
            self.batch_size = batch_size

            # init Two Dense Layer
            self.prev_dense = self.dense_layer(in_size=self.seq_length, out_size=self.rnn_size) # Before Decoder Input.
            self.post_dense = self.dense_layer(in_size=self.rnn_size, out_size=self.seq_length) # After Decoder Output.

            # RESULT
            self.training_decoder_output, self.predicting_decoder_output = self.forward()

        def dense_layer(self, in_size, out_size):
            weight = tf.Variable(tf.random_normal([in_size, out_size])) # shape (in_size, out_size)
            biases = tf.Variable(tf.constant(0.1, shape=[out_size, ])) # shape (out_size, )
            return {'weight':weight, 'biases':biases, }
        
        def forward(self):

            def get_lstm_cell(rnn_size):
                lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return lstm_cell

            def apply_dense(X, p, steps):
                if p == 'prev':
                    X = tf.reshape(X, [-1, self.seq_length]) # [batch_size, steps, input_size] -> [batch_size*steps, input_size]
                    X = tf.matmul(X, self.prev_dense['weight']) + self.prev_dense['biases']
                    X = tf.reshape(X, [-1, steps, self.rnn_size])
                elif p == 'post':
                    X = tf.reshape(X, [-1, self.rnn_size]) # [batch_size, steps, input_size] -> [batch_size*steps, input_size]
                    X = tf.matmul(X, self.post_dense['weight']) + self.post_dense['biases']
                    X = tf.reshape(X, [-1, steps, self.seq_length])
                else:
                    raise ValueError('Wrong argument while using apply_dense()')
                return X
            
            # 2. Biuld up Decoder RNN unit
            self.cells = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(self.rnn_size) for _ in range(self.num_layers)])
             
            # 3-1. Training decoder
            with tf.variable_scope("decode"):

                X = apply_dense(self.decoder_input, 'prev', steps=self.decoder_steps) # Embedding

                train_length = tf.fill([self.batch_size], self.decoder_steps)
                training_decoder_output, _ = tf.nn.dynamic_rnn(self.cells, X, 
                                                    sequence_length=train_length, initial_state=self.encoder_state, dtype=tf.float32)
                
                training_decoder_output = apply_dense(training_decoder_output, 'post', steps=self.decoder_steps) # De-embedding

            # 3-2. Predicting decoder
            with tf.variable_scope("decode", reuse=True):
                # 創建 start_tokens for each batches
                self.start_tokens = tf.fill([batch_size], 0.5, name='start_tokens')
                # 定義只走一步
                PREDICT_STEP = 1
                train_length = tf.fill([self.batch_size], PREDICT_STEP)
                
                # STEP 1
                X = apply_dense(self.start_tokens, 'prev', steps=PREDICT_STEP)  # Embedding
                step1_output, step1_state = tf.nn.dynamic_rnn(self.cells, X, 
                                                    sequence_length=train_length, initial_state=self.encoder_state, dtype=tf.float32)
                X = apply_dense(step1_output, 'post', steps=PREDICT_STEP) # De-embedding

                # STEP 2
                X = apply_dense(X, 'prev', steps=PREDICT_STEP)  # Embedding
                step2_output, step2_state = tf.nn.dynamic_rnn(self.cells, X, 
                                                    sequence_length=train_length, initial_state=step1_state, dtype=tf.float32)
                X = apply_dense(step2_output, 'post', steps=PREDICT_STEP) # De-embedding

                # STEP 3
                X = apply_dense(X, 'prev', steps=PREDICT_STEP)  # Embedding
                predicting_decoder_output, _ = tf.nn.dynamic_rnn(self.cells, X, 
                                                    sequence_length=train_length, initial_state=step2_state, dtype=tf.float32)
                predicting_decoder_output = apply_dense(predicting_decoder_output, 'post', steps=PREDICT_STEP) # De-embedding
                

            return training_decoder_output, predicting_decoder_output

