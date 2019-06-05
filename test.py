
# @commandline_args.txt

import argparse
import sys
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser.add_argument('-if', '--in_frames', type=int, default=20,
                    help='Number of frames in one input sequence (default:20)')
parser.add_argument('-ob', '--out_band', type=int, default=10,
                    help='Number of output-band frames in the mid of input sequence (default:10)')


args = parser.parse_args()
if args.in_frames <= args.out_band:
    parser.error('-if must larger than -ob.')
elif (args.in_frames-args.out_band)%2 != 0:
    parser.error('The value -if larger than -ob must divisible by 2.')