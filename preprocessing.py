# encoding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pandas as pd

from sklearn import preprocessing as P

import matplotlib.pyplot as plt

def _load_data_as_frame(filepath):
	data = pd.read_csv(filepath)
	if _is_missing(data):
		print('Missing data')
	return data


def load_data(filepath):
	data = _load_data_as_frame(filepath)
	return data.as_matrix()


def _is_missing(data):
	return data.isnull().values.any()


def _get_mean_var_atts(data):
	return np.mean(data, axis=0), np.var(data, axis=0)


def _get_max_min_atts(data):
	return np.max(data, axis=0), np.min(data, axis=0)


def get_dist(data):
	mj, mn = _get_max_min_atts(data)
	mean, var = _get_mean_var_atts(data)
	dist = np.concatenate((mj, mn, mean, var), axis=0)
	return dist


def _outliers_iqr(ys):
  quartile_1, quartile_3 = np.percentile(ys, [10, 90])
  iqr = quartile_3 - quartile_1
  lower_bound = quartile_1 - (iqr * 1.5)
  upper_bound = quartile_3 + (iqr * 1.5)
  return np.where((ys > upper_bound) | (ys < lower_bound))


def find_outliers(data, n_atts):
	atts_w_outliers = dict()
	for i in range(n_atts):
		outliers_att = _outliers_iqr(data[i,:])
		if not outliers_att:
			atts_w_outliers[str(i)] = [e for e in outliers_att]
	return atts_w_outliers


def normalize(data):
	return P.scale(data)			


def plot_1d_att(data, att, save=False):
	name = 'Parameter-%s' % str(att + 1)
	plt.cla()
	h = plt.hist(data[:,att], bins=500)
	plt.title('Distribution for attribute %s' % name)
	plt.tight_layout()
	if not save:
		plt.show()
		return
	plt.savefig(name + '.png')
