import tensorflow as tf
import numpy as np
import time
import os,sys
import argparse
import datetime
from matplotlib import pyplot as plt
import importlib
# from hyperopt import fmin, tpe, hp, Trials, partial

import DataProcess
# import cnn_trainer
# import rnn_trainer

parser = argparse.ArgumentParser(description='Skeleton-based action segment RNN model.', fromfile_prefix_chars='@')

# Data parameter
parser.add_argument('-d', '--data_path', type=str, default='data',
                    help='Path to Dataset (default:\"../data\")')
parser.add_argument('-jnum', '--num_joint', type=int, default=18,
                    help='Number of joints (default:18)')
parser.add_argument('-jdim', '--coord_dim', type=int, default=2,
                    help='Dimension of joint coordinate (default:2)')

# Training parameter
parser.add_argument('-e', '--epochs', type=int, default=100, # hyperopt
                    help='Number of Epochs (default:100)')
parser.add_argument('-b', '--batch_size', type=int, default=15, # hyperopt
                    help='Batch Size (default:15)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                    help='Learning Rate (default:0.0001)')
parser.add_argument('--hyperopt', action="store_true",
                    help='Do hyperopt training or not.')

# Display parameter
parser.add_argument('--display_step', type=int, default=50,
                    help='Display loss for every N batches (default:50)')
parser.add_argument('--test', action="store_true",
                    help='Only do inference')
parser.add_argument('--info', type=str,
                    help='Information about this training.')

# Model selection
parser.add_argument('--model', type=str, default="cnn", 
                    help='Choose model \"cnn\" or \"rnn\" (default:cnn)')

# Model structure parameter
parser.add_argument('-rs', '--rnn_size', type=int, default=50, # hyperopt
                    help='RNN Size (default:50)')
parser.add_argument('-rl', '--num_layers', type=int, default=4, # hyperopt
                    help='Number of Layers (default:4)')
parser.add_argument('-ds', '--decoder_steps', type=int, default=3,
                    help='Steps of decoding (default:3)')
parser.add_argument('-if', '--in_frames', type=int, default=20,
                    help='Number of frames in one input sequence (default:20)')
parser.add_argument('-ob', '--out_band', type=int, default=10,
                    help='Number of output-band frames in the mid of input sequence (default:10)')
args = parser.parse_args()

if args.info != None:
    # save args
    with open('./model/command_args.txt', 'w') as f:
        f.write('\n'.join(sys.argv[1:]))
    # save information
    now = datetime.datetime.now()
    current_time = '{:04d}_{:02d}_{:02d}_{:02d}{:02d}{:02d}\n'.format(
        now.year, now.month, now.day, 
        now.hour, now.minute, now.second)
    with open('./model/info.txt', 'w') as out_file:
        out_file.write(current_time)
        out_file.write(args.info)
        out_file.write('Model: {}'.format(args.model))

# Dynamic import pkg
if args.model == "cnn":
    Solver = importlib.import_module('cnn.Solver').Solver
    Model = importlib.import_module('cnn.Resnet').Resnet
    pass
elif args.model == "rnn":

    # check argument
    if args.in_frames <= args.out_band:
        parser.error('-if must larger than -ob.')
    elif (args.in_frames-args.out_band)%2 != 0:
        parser.error('The value -if larger than -ob must divisible by 2.')
    Solver = importlib.import_module('rnn_trainer')
else:
    parser.error('Unkown model. Choose model from \"cnn\" or \"rnn\"')


def main():
    # Input size of each steps
    input_size = args.num_joint*args.coord_dim

    # Loading data
    DataLoader = DataProcess.DataProcess(path=args.data_path, 
                                         batch_size=args.batch_size, 
                                         num_joint=args.num_joint,
                                         coord_dim=args.coord_dim,
                                         # input_size=input_size, 
                                         decoder_steps=args.decoder_steps) # TODO
    # Build graph & Train/Test
    solver = Solver()
    if args.test:
        solver.test(args=args, DataLoader=DataLoader)
    else:
        # Biuld net
        if args.model == "cnn":
            net = Model(name="resnet",
                         layer_n=5,
                         in_shape=[args.in_frames, args.num_joint, args.coord_dim], 
                         out_shape=[args.out_band], 
                         num_steps=args.epochs*DataLoader.Get_num_batch(DataLoader.train_set['source'], args.in_frames),
                         lr=args.learning_rate)
        elif args.model == "rnn":
            net = Model()
        solver.train(args=args, DataLoader=DataLoader, net=net)
        solver.test(args=args, DataLoader=DataLoader)

if __name__ == '__main__':
    main()