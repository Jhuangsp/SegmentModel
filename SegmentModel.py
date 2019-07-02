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
    inputs  = tf.placeholder(tf.float32, [None, None, None], name='inputs')  # (batch_size, steps, input_size)
    targets = tf.placeholder(tf.float32, [None, None, None], name='targets') # (batch_size, steps, output_size)
    
    # Learning rate
    keep_rate = tf.placeholder(tf.float32, name='keep_rate')
    
    return inputs, targets, keep_rate

class Seq2Seq(object):
    '''
    Sequence to Sequence Model
    
    Parameter:
        - input_data:           (tf.placeholder)  padding 完的1個 batch 的輸入資料 (tensor:[batch_size, steps])
        - targets:              (tf.placeholder)  padding 完的1個 batch 的目標資料 (tensor:[batch_size, steps])
        
        - seq_length:       Steps of one input sequence (dtype=int)
        - out_length:       Number of output-band frames in the mid of input sequence (dtype=int)
        - rnn_size:         Size of lstm hiden state
        - num_layers:       Number of multy-layer-lstm layers
        - batch_size:       Batch size

        - input_size:           Input size of each steps
        - decoder_steps:    Number of steps should decoder do.
    '''
    def __init__(self, input_data, targets, keep_rate,
                  seq_length, out_length, rnn_size, num_layers, batch_size, 
                  input_size, decoder_steps):
        super(Seq2Seq, self).__init__()

        self.input_data = input_data
        self.targets = targets
        self.keep_rate = keep_rate

        self.seq_length = seq_length
        self.out_length = out_length
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        # self.batch_size = batch_size
        batch_size_tmp = tf.shape(self.input_data)[0]
        self.batch_size = batch_size_tmp

        self.input_size = input_size
        self.decoder_steps = decoder_steps

        def process_decoder_input(data, batch_size):
            '''
                Add START_TOKEN at the front and remove the LAST_TOKEN. (used for training)
                    e.g. Ground Truth=[4,5,6], then Input=[start_tokens,4,5]
                Parameter:
                    - data: shape=(batch_size, sequence_len)
                Return:
                    - decoder_input: target start with START_TOKEN without LAST_TOKEN
            '''
            # START_TOKEN = np.full((batch_size, 1, output_len), 0.5)
            data_without_end = tf.strided_slice(data, begin=[0, 0], end=[batch_size, -1], strides=[1, 1]) #TODO?
            decoder_input = tf.concat([tf.fill([batch_size, 1, self.out_length], 0.5), data_without_end], axis=1) #TODO?
            return decoder_input


        self.encoder = self.Encoder(self.input_data, self.keep_rate,
                                    self.seq_length, self.rnn_size, self.num_layers, self.batch_size,
                                    self.input_size)

        # Preprocess the target data
        # batch_size_tmp = tf.shape(self.targets)[0]
        # self.decoder_input = process_decoder_input(self.targets, batch_size_tmp)
        self.decoder_input = process_decoder_input(self.targets, self.batch_size)

        self.decoder = self.Decoder(self.encoder.encoder_state,
                                    self.decoder_input,
                                    self.keep_rate,
                                    self.out_length, self.rnn_size, self.num_layers, self.batch_size,
                                    self.decoder_steps)


    class Encoder(object):
        '''
        Parameter:
            - input_data:       Input tensor

            - seq_length:       Steps of one input sequence (dtype=int)
            - rnn_size:         Size of lstm hiden state
            - num_layers:       Number of multy-layer-lstm layers
            - batch_size:       Batch size

            - input_size:       Input size of each steps
        
        Result:
            - encoder_output:   Short-term momory output by ALL steps [batch_size, steps, RNN_size]
            - encoder_state:    Long-term memory and Short-term momory at the end of steps 
                                    ([batch_size, RNN_size], [batch_size, RNN_size])
        '''
        def __init__(self, input_data, keep_rate, seq_length, 
                       rnn_size, num_layers, batch_size, input_size):
            super(Seq2Seq.Encoder, self).__init__()

            self.input_data = input_data
            self.keep_rate = keep_rate

            self.seq_length = seq_length
            self.rnn_size = rnn_size
            self.num_layers = num_layers
            self.batch_size = batch_size
            self.input_size = input_size

            # Init One Dense Layer
            self.encoder_dense = self.dense_layer(in_size=self.input_size, out_size=self.rnn_size) # Before Eecoder Input.

            # LSTM unit
            def get_lstm_cell(rnn_size):
                #lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_size, layer_norm=False, dropout_keep_prob=self.keep_rate)
                #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=0.9, output_keep_prob=0.9) # dropout
                #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=0.9) # dropout
                return lstm_cell

            # Multi-layer LSTM
            self.cells = tf.contrib.rnn.MultiRNNCell([ get_lstm_cell(self.rnn_size) for _ in range(self.num_layers) ])
            #self.cells = tf.nn.rnn_cell.DropoutWrapper(self.cells, output_keep_prob=0.9) # dropout

            # Result
            self.encoder_output, self.encoder_state = self.forward()

        def dense_layer(self, in_size, out_size):
            weight = tf.Variable(tf.random_normal([in_size, out_size])) # shape (in_size, out_size)
            biases = tf.Variable(tf.constant(0.1, shape=[out_size, ])) # shape (out_size, )
            return {'weight':weight, 'biases':biases, }

        def forward(self):
                        
            # Embedding
            X = tf.reshape(self.input_data, [-1, self.input_size]) # [batch_size, steps, input_size] -> [batch_size*steps, input_size]
            X = tf.matmul(X, self.encoder_dense['weight']) + self.encoder_dense['biases']
            self.encoder_embed_input = tf.reshape(X, [-1, self.seq_length, self.rnn_size])

            # Encoder RNN
            # batch_size_tmp = tf.shape(self.encoder_embed_input)[0]
            # sequence_length = tf.fill([batch_size_tmp], self.seq_length) # TODO
            sequence_length = tf.fill([self.batch_size], self.seq_length) # TODO
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

        Parameter：
            - encoder_state:    encoder端輸出的狀態向量 (tensor)
            - decoder_input:    decoder端输入 (tensor) (Only used in training)

            - out_length:       Number of output-band frames in the mid of input sequence (dtype=int)
            - rnn_size:         Size of lstm hiden state
            - num_layers:       Number of multy-layer-lstm layers
            - batch_size:       Batch size

            - decoder_steps:    Number of steps should decoder do.

        Result:
            - training_decoder_output:     Short-term momory output by ALL steps of training [batch_size, steps, RNN_size]
            - predicting_decoder_output:   Short-term momory output by ALL steps of predicting [batch_size, steps, RNN_size]
        '''

        def __init__(self, encoder_state, decoder_input, keep_rate,
                       out_length, rnn_size, num_layers, batch_size, decoder_steps):
            super(Seq2Seq.Decoder, self).__init__()

            self.encoder_state = encoder_state
            self.decoder_input = decoder_input
            self.keep_rate = keep_rate

            self.out_length = out_length
            self.rnn_size = rnn_size
            self.num_layers = num_layers
            self.batch_size = batch_size
            self.decoder_steps = decoder_steps

            # init Two Dense Layer
            self.prev_dense = self.dense_layer(in_size=self.out_length, out_size=self.rnn_size) # Before Decoder Input.
            self.post_dense = self.dense_layer(in_size=self.rnn_size, out_size=self.out_length) # After Decoder Output.

            # LSTM unit
            def get_lstm_cell(rnn_size):
                lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(rnn_size, layer_norm=False, dropout_keep_prob=self.keep_rate)
                # lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=0.9, output_keep_prob=0.9) # dropout
                #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=0.9) # dropout
                return lstm_cell

            # Multi-layer LSTM
            self.cells = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(self.rnn_size) for _ in range(self.num_layers)])
            #self.cells = tf.nn.rnn_cell.DropoutWrapper(self.cells, output_keep_prob=0.9) # dropout

            # RESULT
            self.training_decoder_output, self.predicting_decoder_output = self.forward()

        def dense_layer(self, in_size, out_size):
            weight = tf.Variable(tf.random_normal([in_size, out_size])) # shape (in_size, out_size)
            biases = tf.Variable(tf.constant(0.1, shape=[out_size, ])) # shape (out_size, )
            return {'weight':weight, 'biases':biases, }
        
        def forward(self):

            def apply_dense(X, p, steps):
                if p == 'prev':
                    X = tf.reshape(X, [-1, self.out_length]) # [batch_size, steps, input_size] -> [batch_size*steps, input_size]
                    X = tf.matmul(X, self.prev_dense['weight']) + self.prev_dense['biases']
                    X = tf.reshape(X, [-1, steps, self.rnn_size])
                elif p == 'post':
                    X = tf.reshape(X, [-1, self.rnn_size]) # [batch_size, steps, input_size] -> [batch_size*steps, input_size]
                    X = tf.matmul(X, self.post_dense['weight']) + self.post_dense['biases']
                    X = tf.reshape(X, [-1, steps, self.out_length])
                else:
                    raise ValueError('Wrong argument while using apply_dense()')
                return X
            
            # Training decoder
            with tf.variable_scope("decode"):
                # Embedding
                X = apply_dense(self.decoder_input, 'prev', steps=self.decoder_steps) 
                # Decoder RNN
                train_length = tf.fill([self.batch_size], self.decoder_steps)
                training_decoder_output, _ = tf.nn.dynamic_rnn(self.cells, X, 
                                                    sequence_length=train_length, initial_state=self.encoder_state, dtype=tf.float32)
                # De-embedding
                training_decoder_output = tf.sigmoid(apply_dense(training_decoder_output, 'post', steps=self.decoder_steps))
                # training_decoder_output = apply_dense(training_decoder_output, 'post', steps=self.decoder_steps)


            # Predicting decoder
            with tf.variable_scope("decode", reuse=True):
                # step = 1
                PREDICT_STEP = 1
                # Create start_tokens for each batches [batch_size, steps, output_len]
                self.start_tokens = tf.fill([self.batch_size, 1, self.out_length], 0.5, name='start_tokens')
                train_length = tf.fill([self.batch_size], PREDICT_STEP)
                
                # STEP 1
                X = apply_dense(self.start_tokens, 'prev', steps=PREDICT_STEP)  # Embedding
                step1_output, step1_state = tf.nn.dynamic_rnn(self.cells, X, 
                                                    sequence_length=train_length, initial_state=self.encoder_state, dtype=tf.float32)
                X = tf.sigmoid(apply_dense(step1_output, 'post', steps=PREDICT_STEP)) # De-embedding
                # X = apply_dense(step1_output, 'post', steps=PREDICT_STEP) # De-embedding

                # STEP 2
                X = apply_dense(X, 'prev', steps=PREDICT_STEP)  # Embedding
                step2_output, step2_state = tf.nn.dynamic_rnn(self.cells, X, 
                                                    sequence_length=train_length, initial_state=step1_state, dtype=tf.float32)
                X = tf.sigmoid(apply_dense(step2_output, 'post', steps=PREDICT_STEP)) # De-embedding
                # X = apply_dense(step2_output, 'post', steps=PREDICT_STEP) # De-embedding

                # STEP 3
                X = apply_dense(X, 'prev', steps=PREDICT_STEP)  # Embedding
                predicting_decoder_output, _ = tf.nn.dynamic_rnn(self.cells, X, 
                                                    sequence_length=train_length, initial_state=step2_state, dtype=tf.float32)
                predicting_decoder_output = tf.sigmoid(apply_dense(predicting_decoder_output, 'post', steps=PREDICT_STEP)) # De-embedding
                # predicting_decoder_output = apply_dense(predicting_decoder_output, 'post', steps=PREDICT_STEP) # De-embedding
                

            return training_decoder_output, predicting_decoder_output

