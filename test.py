
# @commandline_args.txt

import argparse
import sys
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--foo')
parser.add_argument('bar', nargs='?')

print(parser.parse_args())