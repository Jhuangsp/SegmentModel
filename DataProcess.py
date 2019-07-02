import numpy as np
import os
import pprint
import glob
import json
import random
import matplotlib.pyplot as plt

import utils

pp = pprint.PrettyPrinter(indent=4)

class DataProcess(object):
    """docstring for DataProcess"""
    def __init__(self, path, batch_size, input_size, decoder_steps):
        super(DataProcess, self).__init__()

        self.source_path = {} # {'activity_name':['frame1_path', 'frame2_path', ..., 'frameEND_path'], 'activity2_name':[...], ...}
        print('Skeleton file:')
        self.max_len = 0
        for activity in glob.glob(os.path.join(path, 'skeleton', '*')):
            activity_name = activity.split(os.sep)[-1]
            self.source_path[activity_name] = glob.glob(os.path.join(activity, '*.json')) 
            print('Activity {:<25s} has {:>4d} frames.'.format(activity_name, len(self.source_path[activity_name])))
            self.max_len = len(self.source_path[activity_name]) if len(self.source_path[activity_name]) > self.max_len else self.max_len

        self.target_path = {}
        for activity in glob.glob(os.path.join(path, 'label', '*.npy')):
            activity_name = activity.split(os.sep)[-1].split('.')[0]
            self.target_path[activity_name] = activity
        print('Label file:')
        pp.pprint(self.target_path)

        self.batch_size = batch_size
        self.input_size = input_size
        self.decoder_steps = decoder_steps

        # Loading data
        self.source_data, self.target_data = self.Load_data()
        # Split the Training/Validation data
        self.train_set, self.valid_set = self.Split_dataset()

    def Load_data(self):
        '''
        Loading data

        return: 
         - source_data: [samples, self.max_len(steps), input_size]
         - target_data: [samples, decoder_steps, self.max_len]
        '''

        # Dick type
        print('==> Loading data...\n')
        source_data = {}
        print('Source data shape (samples, max_data_steps, input_size):') 
        for activity_name, frames_list in self.source_path.items():
            frame_data = np.zeros((len(frames_list) ,self.input_size), dtype=np.float32)
            for i,frame in enumerate(frames_list):
                with open(frame) as json_file:
                    data = json.load(json_file)
                    frame_data[i] = utils.normalize(np.array(data['people'][0]['pose_keypoints']))
            source_data[activity_name] = np.copy(frame_data)
            print('  sample {}: {}'.format(activity_name, source_data[activity_name].shape))


        target_data = {}
        print('Target data shape (samples, decoder_steps, max_data_steps):') 
        for activity_name, label_list in self.target_path.items():
            loaded = np.load(label_list)
            lf = np.zeros((self.decoder_steps, loaded.shape[0]), dtype=np.float32)
            # ns = [2, 1, 0.25] # less accurate -> more accurate
            # ns = [2, 1, 0.5] # less accurate -> more accurate
            ns = [2, 2, 2] # less accurate -> more accurate
            assert len(ns)==self.decoder_steps, 'length of \'ns\' != \'decoder_steps\'.'
            for s in range(self.decoder_steps):
                lf[s] = utils.gaussian_weighted(loaded, ns[s])
            target_data[activity_name] = np.copy(lf)
            print('  sample {}: {}'.format(activity_name, target_data[activity_name].shape))
        print('\n')

        return source_data, target_data

    def Split_dataset(self):
        '''
        Split data into Training/Validation data
        '''

        print('==> Spliting Dataset...\n')
        # 1 video for validation
        # train_source = dict(list(self.source_data.items())[1:])
        # train_target = dict(list(self.target_data.items())[1:])
        # valid_source = dict(list(self.source_data.items())[:1])
        # valid_target = dict(list(self.target_data.items())[:1])
        train_source = dict(list(self.source_data.items())[:6] + list(self.source_data.items())[7:])
        train_target = dict(list(self.target_data.items())[:6] + list(self.target_data.items())[7:])
        valid_source = dict(list(self.source_data.items())[6:7])
        valid_target = dict(list(self.target_data.items())[6:7])

        print('Training source:')
        for k,i in train_source.items():
            print(' sample {}: {}'.format(k, i.shape))
        print('Training target:')
        for k,i in train_target.items():
            print(' sample {}: {}'.format(k, i.shape))
        print('Validate source:')
        for k,i in valid_source.items():
            print('  sample {}: {}'.format(k, i.shape))
        print('Validate target:')
        for k,i in valid_target.items():
            print('sample {}: {}'.format(k, i.shape))

        train_set = {'source':train_source, 'target':train_target}
        valid_set = {'source':valid_source, 'target':valid_target}
        print('Done Spliting...')
        print('\n')

        return train_set, valid_set

    def Batch_Generator_legacy(self, source_split, target_split, input_seq_length, out_band_length):
        '''
        Define generator to generate batch
        Random pick sample and start point
        
        Return:
        - batch
        '''
        samples = len(source_split)
        start_range = self.max_len - input_seq_length

        num_batch = (samples*start_range)//(input_seq_length*self.batch_size)

        print(' - Total Batches: {:2d} | Videos: {:2d} | Input Length: {:2d}'.format(num_batch, samples, input_seq_length))
        print(' - Batche Shape: ({0:}, {1:}, {2:}) (source) | ({0:}, {3:}, {1:}) (target)\n'.format(
                                                self.batch_size, input_seq_length, self.input_size, self.decoder_steps))

        for batch_i in range(num_batch):
            # random choose sample
            sample_i = np.random.randint(samples, size=self.batch_size)
            # start point
            start_i = np.random.randint(start_range, size=self.batch_size)

            # cut batch [r[i][st:st+2] for (i,st) in enumerate(start)]
            random_targets = [target_split[s] for s in sample_i]
            random_sources = [source_split[s] for s in sample_i]
            targets_batch = [random_targets[i][:,st:st+input_seq_length] for (i,st) in enumerate(start_i)]
            sources_batch = [random_sources[i][st:st+input_seq_length] for (i,st) in enumerate(start_i)]
            
            yield np.array(sources_batch), np.array(targets_batch), num_batch

    def Batch_Generator(self, source_split, target_split, input_seq_length, out_band_length, training=True):
        '''
        Define generator to generate batch
        Random pick sample and start point
        
        Return:
        - batch
        '''
        samples = len(source_split)
        pool = []
        for activity,data in source_split.items():
            for start_i in range(data.shape[0]-input_seq_length):
                pool.append((activity, start_i))
        num_batch = len(pool) // self.batch_size

        OutOfBand_size = (input_seq_length - out_band_length) // 2

        if training:
            random.shuffle(pool)
            print(' - Total Batches: {:2d} | Videos: {:2d} | Input Length: {:2d}'.format(num_batch, samples, input_seq_length))
            print(' - Batche Shape: ({0:}, {1:}, {2:}) (source) | ({0:}, {3:}, {1:}) (target)\n'.format(
                                                    self.batch_size, input_seq_length, self.input_size, self.decoder_steps))

        for batch_i in range(num_batch):
            pick = pool[batch_i*self.batch_size:(batch_i+1)*self.batch_size]
            # random choose sample & start point
            sample_i, start_i = zip(*pick)

            # cut batch [r[i][st:st+2] for (i,st) in enumerate(start)]
            random_targets = [target_split[s] for s in sample_i]
            random_sources = [source_split[s] for s in sample_i]
            targets_batch = [random_targets[i][:,st+OutOfBand_size:st+input_seq_length-OutOfBand_size] for (i,st) in enumerate(start_i)]
            sources_batch = [random_sources[i][st:st+input_seq_length] for (i,st) in enumerate(start_i)]
            
            yield np.array(sources_batch), np.array(targets_batch), num_batch

