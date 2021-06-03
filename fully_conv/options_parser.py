import argparse


def setup_options():
    options = argparse.ArgumentParser()
    options.add_argument('-r', action='store', dest='seed',
                         default=3, type=int)
    options.add_argument('-s', action='store', dest='size',
                         default=8, type=int)
    options.add_argument('--model', action='store', dest='model',
                         default='conv', type=str)
    options.add_argument('--num_classes', action= 'store', dest='num_classes', default=10, type=int)
    options.add_argument('--width', action='store', dest='width', default=16, type=int)
    options.add_argument('--trials', action='store', dest='trials', default=1, type=int)
    options.add_argument('--run_idx', action='store', dest='run_idx', default=0, type=int)
    return options.parse_args()

