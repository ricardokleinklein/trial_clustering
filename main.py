# encoding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import string
import numpy as np

from docopt import docopt
from tqdm import tqdm

from utils import *

if __name__ == "__main__":
	path = '/Users/ricardokleinlein/Desktop/py2/data/data.csv'
	data = load_data(path)
	

