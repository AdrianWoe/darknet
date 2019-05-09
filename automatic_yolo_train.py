import argparse
import sys
from subprocess import call

# initiate the parser
parser = argparse.ArgumentParser()
# add long and short argument
parser.add_argument("--gpu", help="index of the gpu to use")
parser.add_argument("--iter", help="Name of the yolo iteration")
# read arguments from the command line
args = parser.parse_args()
if not args.iter:
    sys.exit("No GPU index given")
if not args.iter:
    sys.exit("No iteration name given")

gpu_index = args.gpu
iter_name = args.iter

command = 'CUDA_VISIBLE_DEVICES={} ./darknet detector train {}/{}.data {}/{}_train.cfg darknet53.conv.74 -map -dont_show | tee logs/{}_train.log'.format(
    gpu_index, iter_name, iter_name, iter_name, iter_name, iter_name)
print(command)

call(
    command, shell=True)
