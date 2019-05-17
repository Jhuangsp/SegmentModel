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
    
    - source_vocab_size:                        input char to int 的字典大小(長度)
    - target_vocab_size:                        target char to int 的字典大小(長度)
    
    - encoding_embedding_size:                   input embedding 後的vector大小(維度) removed replaced by rnnsize
    - decoding_embedding_size:                   output de-embedding 的vector大小(維度) removed
    
    - rnn_size:                                 RNN hiden state 的維度 (cells數)
    - num_layers:                               RNN的層數
    - input_size:                               Input size of each steps
    '''
    def __init__(self, 
                  input_data, targets, lr, source_seq_len, target_seq_len, max_target_seq_len,
                  dataloader,
                  rnn_size, num_layers, batch_size):
        super(Seq2Seq, self).__init__()

        self.input_data = input_data
        self.targets = targets
        self.lr = lr
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.max_target_seq_len = max_target_seq_len

        self.dataloader = dataloader
        self.source_vocab_size = len(self.dataloader.source_dict['letter2int'])
        self.target_vocab_size = len(self.dataloader.target_dict['letter2int'])

        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size

        # Preprocess the target data
        def process_decoder_input(data, vocab_to_int, batch_size):
            '''
            Add <GO> and remove the last char
            '''
            # 刪去最後一個字符
            ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

            return decoder_input


        self.encoder = self.Encoder(self.input_data, 
                                    self.source_seq_len,
                                    self.source_vocab_size, 
                                    self.rnn_size, 
                                    self.num_layers,
                                    self.input_size)

        # Preprocess the target data
        self.decoder_input = process_decoder_input(self.targets, 
                                            self.dataloader.target_dict['letter2int'], 
                                            self.batch_size)

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
        Parameter：
        - input_data:               輸入 tensor
        - source_seq_len:   Source data 的序列原始長度 (tensor:[batch_size], dtype=int)

        - source_vocab_size:        Source data 的 Mapping dictionary 大小

        - rnn_size:                 RNN hiden state 的維度
        - num_layers:               RNN的層數
        
        輸出:
        - encoder_output:           所有steps的短期記憶 h 輸出 [batch_size, steps, RNN_size]
        - encoder_state:            包含兩部分 (最後結束時的長期記憶 c, 最後一次輸出的短期記憶 h) ([batch_size, RNN_size], [batch_size, RNN_size])
            rnn_size: 
              Tensorflow’s num_units is the size of the LSTM’s hidden state 
              (which is also the size of the output if no projection is used). 
              To make the name num_units more intuitive, you can think of it as the 
              number of hidden units in the LSTM cell, or the number of memory units in the cell.
              (https://www.quora.com/What-is-the-meaning-of-%E2%80%9CThe-number-of-units-in-the-LSTM-cell)
              (https://stackoverflow.com/questions/37901047/what-is-num-units-in-tensorflow-basiclstmcell)
              我理解為 memory 的維度

        '''
        def __init__(self, input_data, source_seq_len, 
                       source_vocab_size, 
                       rnn_size, num_layers, input_size):
            super(Seq2Seq.Encoder, self).__init__()

            self.input_data = input_data
            self.source_seq_len = source_seq_len

            self.source_vocab_size = source_vocab_size

            self.rnn_size = rnn_size
            self.num_layers = num_layers
            self.input_size = input_size

            # 輸出
            self.encoder_output, self.encoder_state = self.forward()

        def forward(self):
            # RNN cell
            def get_lstm_cell(rnn_size):
                lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return lstm_cell
                        
            # Embedding Weight
            # shape (input_size_of_each_step, embed_size)
            embedding_weight = tf.Variable(tf.random_normal([self.input_size, self.rnn_size]))
            # shape (embed_size, )
            embedding_biases = tf.Variable(tf.constant(0.1, shape=[self.rnn_size, ]))


            # Multi-layer LSTM
            self.cells = tf.contrib.rnn.MultiRNNCell([ get_lstm_cell(self.rnn_size) for _ in range(self.num_layers) ])


            # Encoder embedding [shape=(?, ?, self.encoding_embedding_size), dtype=float32]
            X = tf.reshape(self.input_data, [-1, self.input_size]) # [batch_size, steps, input_size] -> [batch_size*steps, input_size]
            self.encoder_embed_input = tf.matmul(X, embedding_weight) + embedding_biases
            
            # tf.nn.dynamic_rnn(lstm_cells, input, sequence_length=steps, dtype=tf.float32)
            encoder_output, encoder_state = tf.nn.dynamic_rnn(self.cells, self.encoder_embed_input, 
                                                              sequence_length=self.source_seq_len, dtype=tf.float32)
            
            return encoder_output, encoder_state 

    class Decoder(object):
        '''
        Decoder Layer
        
        Parameter：
        - encoder_state: encoder端輸出的狀態向量 (tensor)
        - decoder_input: decoder端输入 (tensor)
        - target_seq_len: target數據序列的長度 (tensor)
        - max_target_seq_len: target數據序列的最大長度 (tensor)

        - target_letter_to_int: target數據的映射表 (dict)

        - rnn_size: RNN hiden state 的維度
        - num_layers: RNN的層數
        - batch_size: batch size
        '''

        def __init__(self, encoder_state, decoder_input,
                       target_seq_len, max_target_seq_len,
                       target_letter_to_int, 
                       rnn_size, num_layers, batch_size):
            super(Seq2Seq.Decoder, self).__init__()

            self.encoder_state = encoder_state
            self.decoder_input = decoder_input

            self.target_letter_to_int = target_letter_to_int
            self.num_layers = num_layers
            self.rnn_size = rnn_size
            self.batch_size = batch_size
            self.target_seq_len = target_seq_len
            self.max_target_seq_len = max_target_seq_len

            # 輸出
            self.training_decoder_output, self.predicting_decoder_output = self.forward()
            
        def forward(self):

            def prepare_helper_input(output_length, embedding_size, embed_data):
                # Decoder Embedding
                ## Biuld up embedding map
                embedding_map = tf.Variable(tf.random_uniform([output_length, embedding_size])) 
                ## Do target embedding
                un_embedd_data = tf.nn.embedding_lookup(embedding_map, embed_data)
                
                return embedding_map, un_embedd_data

            def get_lstm_cell(rnn_size):
                lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return lstm_cell

            # 1. Decoder Embedding
            ( self.embedding_map, 
              self.decoder_embed_input ) = prepare_helper_input(
                                                    self.output_length,
                                                    self.rnn_size,
                                                    self.decoder_input)
            
            # 2. Biuld up Decoder RNN unit
            self.cells = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(self.rnn_size) for _ in range(self.num_layers)])
             
            # 3-1. Training decoder
            with tf.variable_scope("decode"):
                self.training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=self.decoder_embed_input,
                                                                    sequence_length=self.target_seq_len,
                                                                    time_major=False)
                # 建立 Decoder
                self.training_decoder = tf.contrib.seq2seq.BasicDecoder(self.cells,
                                                                   self.training_helper,
                                                                   self.encoder_state) 
                
                # 對 decoder執行 dynamic decoding。透過 maximum_iterations參數定義最大序列長度。
                training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(self.training_decoder,
                                                                                  impute_finished=True,
                                                                                  maximum_iterations=self.max_target_seq_len)
                
            # 3-2. Predicting decoder
            # 與 Training decoder共享參數
            with tf.variable_scope("decode", reuse=True):
                # 創建一個常數 start_tokens tensor並複製為 batch_size的大小
                self.start_tokens = tf.tile(tf.constant([self.target_letter_to_int['<GO>']], dtype=tf.int32), [self.batch_size], name='start_tokens')
                
                self.predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding_map,
                                                                        start_tokens=self.start_tokens,
                                                                        end_token=self.end_output)
                self.predicting_decoder = tf.contrib.seq2seq.BasicDecoder(self.cells,
                                                                self.predicting_helper,
                                                                self.encoder_state)
                predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(self.predicting_decoder,
                                                                                    impute_finished=True,
                                                                                    maximum_iterations=self.max_target_seq_len)
            
            return training_decoder_output, predicting_decoder_output


