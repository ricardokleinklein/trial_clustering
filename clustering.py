# encoding: utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import sklearn as sk

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score as SC

from hparams import hparams

def kmeans(X, hparams):
	km = KMeans(hparams.n_clusters)
	km.fit(X)
	return km.labels_, km.cluster_centers_


def gmm(X, hparams):
	gmm = GMM(hparams.n_clusters)
	gmm.fit(X)
	return gmm.predict(X), gmm.means_


def measure_silhouette(labels):
	raise NotImplementedError


def get_char(labels):
	return [chr(97 + l) for l in labels]






