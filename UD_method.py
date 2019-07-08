import numpy as np
import os
import argparse
from matplotlib import pyplot as plt

import DataProcess

parser = argparse.ArgumentParser(description='Skeleton-based action segment RNN model.', fromfile_prefix_chars='@')

# Data parameter
parser.add_argument('-d', '--data_path', type=str, default='data',
                    help='Path to Dataset (default:\"../data\")')
parser.add_argument('-b', '--batch_size', type=int, default=15, # hyperopt
                    help='Batch Size (default:15)')
parser.add_argument('-jnum', '--num_joint', type=int, default=18,
                    help='Number of joints (default:18)')
parser.add_argument('-jdim', '--coord_dim', type=int, default=2,
                    help='Dimension of joint coordinate (default:2)')
parser.add_argument('-ds', '--decoder_steps', type=int, default=3,
                    help='Steps of decoding (default:3)')


parser.add_argument('-dg', '--degree', type=int, default=1,
                    help='Degree of Derivative (default:1)')

args = parser.parse_args()

if args.degree > 2 or args.degree <= 0:
    parser.error('-dg/--degree should not > 2 or <= 0')

if __name__ == '__main__':

    h = 1.

    if args.degree == 1:
        # First Derivative
        A = np.array([[      1,      1,      1,      1,      1],
                      [   -2*h,     -h,      0,      h,    2*h],
                      [ 4*h**2,   h**2,      0,   h**2, 4*h**2],
                      [-8*h**3,  -h**3,      0,   h**3, 8*h**3],
                      [16*h**4,   h**4,      0,   h**4,16*h**4]])
        B = np.array([0, 1, 0, 0, 0])
        X = np.linalg.solve(A, B)
    else:
        # Second Derivative
        A = np.array([[      1,      1,      1,      1,      1],
                      [   -2*h,     -h,      0,      h,    2*h],
                      [ 4*h**2,   h**2,      0,   h**2, 4*h**2],
                      [-8*h**3,  -h**3,      0,   h**3, 8*h**3],
                      [16*h**4,   h**4,      0,   h**4,16*h**4]])
        B = np.array([0, 0, 2, 0, 0])
        X = np.linalg.solve(A, B)


    # Loading data
    DataLoader = DataProcess.DataProcess(path=args.data_path, 
                                         batch_size=args.batch_size, 
                                         num_joint=args.num_joint,
                                         coord_dim=args.coord_dim,
                                         decoder_steps=args.decoder_steps) # TODO
    
    print(DataLoader.valid_set['source']['squat_front'].shape)
    data = DataLoader.valid_set['source']['squat_front'][:,0,1]
    result = np.zeros((len(data)-4,), np.float32)
    print(data.shape, result.shape)

    for i in range(len(result)):
        mini_data = data[i:i+5]
        # Curve fitting
        # A.T * A * X = A.T * B
        CA = np.array([[1, -2, (-2)**2],
                       [1, -1, (-1)**2],
                       [1,  0, ( 0)**2],
                       [1,  1, ( 1)**2],
                       [1,  2, ( 2)**2]])
        CB = CA.T @ mini_data
        CAA = CA.T @ CA

        CX = np.linalg.solve(CAA, CB)
        mini_data = CA@CX
        result[i] = mini_data@X

    plt.plot(result)
    ls = np.arange(result.shape[0])
    plt.scatter(ls, result)
    plt.plot([0,300], [0,0], color='red')
    plt.xlabel('Frame')
    plt.ylabel('Probability of Changing Point Frame')
    plt.ylim([-0.05, 0.05])
    plt.show()