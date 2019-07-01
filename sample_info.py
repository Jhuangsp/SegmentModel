
import time
import datetime
import os,sys
import argparse

parser = argparse.ArgumentParser(description='TEST', fromfile_prefix_chars='@')

parser.add_argument('-info', '--information', type=str,
                    help='Path to Dataset (default:\"../data\")')

args = parser.parse_args()

now = datetime.datetime.now()
current_time = '{:04d}_{:02d}_{:02d}_{:02d}{:02d}{:02d}\n'.format(
    now.year, 
    now.month, 
    now.day, 
    now.hour, 
    now.minute, 
    now.second)
with open('./info.txt', 'w') as out_file:
    out_file.write(current_time)
    out_file.write(args.information)