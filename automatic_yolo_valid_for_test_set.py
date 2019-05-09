import argparse
import fileinput
import os
import shutil
import sys
from pathlib import Path
from subprocess import call


def create_directory(directory):
    success = False
    count = 10
    while not success and count > 0:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
            # Would be printed in iterX_test.data
            # print('Could not delete directory: %s' % directory)

        try:
            os.mkdir(directory)
            success = True
        except PermissionError:
            # Would be printed in iterX_test.data
            # print('Access denied to directory: {}'.format(directory))
            success = False
        count -= 1


def create_directory_if_nonexisting(directory):
    path_to_dir = Path(directory)
    if not path_to_dir.is_dir():
        success = False
        count = 10
        while not success and count > 0:
            try:
                os.mkdir(directory)
                success = True
            except PermissionError:
                print('Access denied to directory: {}'.format(directory))
                success = False
            count -= 1


# initiate the parser
parser = argparse.ArgumentParser()
# add long and short argument
parser.add_argument("--gpu", help="index of the gpu to use")
parser.add_argument("--iter", nargs='+', help="Name of the yolo iteration")
parser.add_argument("--weights_dir", help="Directory where the weights are stored")
args = parser.parse_args()
if not args.iter:
    sys.exit("No GPU index given")
if not args.iter:
    sys.exit("No iteration name given")
if not args.weights_dir:
    sys.exit("No directory for the weights given")

gpu_index = args.gpu
iter_names = args.iter
weights_dir = args.weights_dir

for iter_name in iter_names:

    weight_names = [name for name in os.listdir(weights_dir) if iter_name in name]

    # Uses the txt which is set at val= in iterX_test.data.
    target_root_dir = os.path.join('results', '{}_test'.format(iter_name))
    create_directory_if_nonexisting(target_root_dir)

    for weight_name in weight_names:
        # The results dir in the data file must be changed for this specific weight.
        # Otherwise every result would override its predecessor

        for line in fileinput.input(os.path.join(iter_name, '{}_test.data'.format(iter_name)), inplace=True):
            if line.startswith('results'):
                step = weight_name.split('.')[0]
                step = step.split('_')[-1]
                target_dir = os.path.join(target_root_dir, step)
                create_directory(target_dir)
                print('results = {}/'.format(target_dir))
            else:
                print(line.rstrip('\n'))

        command = 'CUDA_VISIBLE_DEVICES={} ./darknet detector valid {}/{}_test.data {}/{}_test.cfg {}/{} -dont_show'.format(
            gpu_index, iter_name, iter_name, iter_name, iter_name, weights_dir, weight_name
        )
        print(command)
        call(command, shell=True)
