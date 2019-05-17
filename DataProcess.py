import numpy as np
import os
import pprint
import glob
import json
import matplotlib.pyplot as plt

import utils

pp = pprint.PrettyPrinter(indent=4)

class DataProcess(object):
    """docstring for DataProcess"""
    def __init__(self, path, batch_size, input_size, decoder_steps):
        super(DataProcess, self).__init__()

        self.source_path = {}
        print('Skeleton file:')
        self.max_len = 0
        for i,activity in enumerate(glob.glob(os.path.join(path, 'skeleton', '*'))):
            self.source_path[i] = glob.glob(os.path.join(activity, '*.json')) 
            print('Activity {:<25s} has {:>4d} frames.'.format(activity.split(os.sep)[-1], len(self.source_path[i])))
            self.max_len = len(self.source_path[i]) if len(self.source_path[i]) > self.max_len else self.max_len

        self.target_path = glob.glob(os.path.join(path, 'label', '*.npy'))
        print('Label file:')
        pp.pprint(self.target_path)

        self.batch_size = batch_size
        self.input_size = input_size
        self.decoder_steps = decoder_steps

        # Loading data
        self.source_data, self.target_data = self.Load_data()
        # Split the Training/Validation data
        self.train_set, self.valid_set = self.Split_dataset()
        # Create Batch Generator, use next() to get the 1 batch iteratively
        '''
        self.train_batch_generator = self.Batch_Generator(
                                            self.train_set['source'], 
                                            self.train_set['target'])
        self.valid_batch_generator = self.Batch_Generator(
                                            self.valid_set['source'], 
                                            self.valid_set['target'])
        '''

    def Load_data(self):
        '''
        Loading data

        return: 
         - source_data: [samples, self.max_len(steps), input_size]
         - target_data: [samples, decoder_steps, self.max_len]
        '''

        print('==> Loading data...\n')
        source_data = np.zeros((len(self.source_path), self.max_len ,self.input_size), dtype=np.float64)
        for idxt, (activity, frames) in enumerate(self.source_path.items()):
            frame_data = np.zeros((self.max_len ,self.input_size), dtype=np.float64)
            for i,frame in enumerate(frames):
                with open(frame) as json_file:
                    data = json.load(json_file)
                    #np.array(data['people'][0]['pose_keypoints']).reshape(-1,3)
                    frame_data[i] = utils.normalize(np.array(data['people'][0]['pose_keypoints'])) # need normalize
            source_data[idxt] = frame_data
        print('Source data shape (samples, max_data_steps, input_size):', source_data.shape) 


        target_data = np.zeros((len(self.target_path), self.decoder_steps, self.max_len), dtype=np.float64)
        for idxt, label_file in enumerate(self.target_path):
            loaded = np.load(label_file)
            lf = np.zeros((self.decoder_steps, loaded.shape[0]), dtype=np.float64)
            ns = [2, 1, 0.25] # less accurate -> more accurate
            assert len(ns)==self.decoder_steps, 'length of \'ns\' != \'decoder_steps\'.'
            for s in range(self.decoder_steps):
                lf[s] = utils.gaussian_weighted(loaded, ns[s]) # need blur
            target_data[idxt, :, :lf.shape[1]] = lf
        print('Target data shape (samples, decoder_steps, max_data_steps):', target_data.shape) 
        print('\n')

        return source_data, target_data

    def Split_dataset(self):
        '''
        將資料分割成 Training/Validation data
        '''

        print('==> Spliting Dataset...\n')
        # 將 dataset分割為 train和 1個 batch的 validation
        train_source = self.source_data[self.batch_size:,:,:]
        train_target = self.target_data[self.batch_size:,:,:]
        valid_source = self.source_data[:self.batch_size,:,:]
        valid_target = self.target_data[:self.batch_size,:,:]

        train_set = {'source':train_source, 'target':train_target}
        valid_set = {'source':valid_source, 'target':valid_target}
        print('Done Spliting...')
        print('\n')

        return train_set, valid_set

    def Batch_Generator(self, source_split, target_split):
        '''
        定義generator，用來生成batch, 此步驟同時做padding
        
        回傳:
        - padding完的batch (list, shape=[batch_size, Max_lenght_of_sentence], dtype=int)
        - 原本的長度紀錄 (list, shape=[batch_size], dtype=int)
        '''

        num_batchs = len(source_split)//self.batch_size

        for batch_i in range(0, num_batchs):
            # batch 的開始點
            start_i = batch_i * self.batch_size
            # 切 batch
            sources_batch = source_split[start_i:start_i + self.batch_size]
            targets_batch = target_split[start_i:start_i + self.batch_size]
            
            yield targets_batch, sources_batch

