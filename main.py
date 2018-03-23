# encoding: utf-8
"""
Clustering of a datafile with no previos knowledge about it
but the number of attributes.

Usage:
	main.py <data_dir> <dst_dir>

options:
	-n, --n_atts			Number of attributes (default:15)
	-h, --help				Display help
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
from docopt import docopt
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from preprocessing import *
from features import *
from clustering import *

from hparams import hparams


def get_hparams(hparams):
	return namedtuple('hparams',hparams.keys())(*hparams.values())


if __name__ == "__main__":
	args = docopt(__doc__)
	data_dir = args["<data_dir>"]
	dst_dir = args["<dst_dir>"]

	hparams = get_hparams(hparams)

	data = load_data(data_dir)
	X = normalize(data)
