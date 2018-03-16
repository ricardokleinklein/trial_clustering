# encoding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import sklearn as sk

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tqdm import tqdm

from hparams import hparams

class autoencoder(nn.Module):
	"""simple autoencoder with 1 hidden layer."""

	def __init__(self, in_size, h_size):
		super(autoencoder, self).__init__()

		self.input = nn.Linear(in_size, h_size)
		self.hidden = nn.Linear(h_size, in_size)

	def forward(self, x):
		x = self.input(x)
		x = F.relu(x)
		x = self.hidden(x)
		x = F.relu(x)
		return x

def run_autoencoder(X, net, hparams):
	raise NotImplementedError


def run_pca(X, hparams):
	pca = PCA(hparams.pca_dims)
	pca.fit(X)
	pca_axis = pca.components_
	proj_X = pca.transform(X)
	return pca.components_, proj_X
