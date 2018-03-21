# encoding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import sklearn as sk

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from hparams import hparams


def writedata(X, filename):
	np.savetxt(filename, X)

class netDataset(Dataset):
	"""Pytorch accesible dataset."""
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx,:]


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


def run_autoencoder(X, net, hparams, dst_dir):
	filename = os.path.join(dst_dir, 'encoder_%s_neurons.csv')
	criterion = nn.MSELoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	X = netDataset(X)
	loader = torch.utils.data.DataLoader(X, batch_size=hparams.batch_size,
                                          shuffle=True, num_workers=2)

	for epoch in tqdm(range(hparams.epochs)):  # loop over the dataset multiple times
		running_loss = 0.0
		for i, data in enumerate(loader, 0):
			# get the inputs
			data = data.type(torch.FloatTensor)
			inputs = data
			labels = data
			# wrap them in variable
			inputs, labels = Variable(inputs), Variable(labels)
			# zero the parameter gradients
			optimizer.zero_grad()
			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			# printing stats
			running_loss += loss.data[0]
			if i % 10000 == 9999:
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 10000))
				running_loss = 0.0
	print('Finished training. Saving encoder features in %s' % filename)


def run_pca(X, hparams):
	pca = PCA(hparams.pca_dims)
	pca.fit(X)
	pca_axis = pca.components_
	proj_X = pca.transform(X)
	return pca.components_, proj_X


def scatter2d(X):
	assert X.shape[1] == 2
	plt.scatter(X[:,0], X[:, 1])
	plt.show()


def scatter3d(X, pca=None):
	assert X.shape[1] == 3
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(X[:,0], X[:,1], X[:,2])
	if pca is not None:
		pass
	plt.show()


def run_lda(X, hparams):
	# lda = LDA(n_components=hparams.pca_dims)
	# lda.fit(X)
	# X_p = lda.transform(X)
	# return X_p
	raise NotImplementedError


def run_ica(X, hparams):
	ica = FastICA(hparams.pca_dims)
	ica.fit(X)
	X_p = ica.transform(X)
	return X_p


def run_nmf(X, hparams):
	nmf = NMF(hparams.pca_dims)
	nmf.fit(X)
	X_p = nmf.transform(x)
	return X_p

