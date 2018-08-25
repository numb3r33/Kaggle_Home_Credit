import pandas as pd
import numpy as np
import gc
import feather

from utils import *

from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import log_loss

from sklearn.preprocessing import scale
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors

SEED = 1231
np.random.seed(SEED)


class NearestNeighborsFeatures(BaseEstimator, ClassifierMixin):
	def __init__(self, n_neighbors, metric, k_list, eps=1e-6, n_jobs=1):
		self.n_neighbors = n_neighbors
		self.metric      = metric
		self.eps         = eps
		self.k_list      = k_list
		self.n_jobs      = n_jobs

	def fit(self, X, y=None):
		self.NN = NearestNeighbors(n_neighbors=self.n_neighbors,
									metric=self.metric,
									n_jobs=1,
									algorithm='brute' if self.metric == 'cosine' else 'auto'
									)

		self.NN.fit(X)

		# store target
		self.y_train = y

	def predict(self, X):
		if self.n_jobs == 1:
			test_feats = []

			for i in range(X.shape[0]):
				test_feats.append(self.get_features_for_one(X[i:i+1]))
		else:
			test_feats = Parallel(self.n_jobs)(delayed(self.get_features_for_one)(X[i:i+1]) for i in range(X.shape[0])) 

		return np.vstack(test_feats)

	def get_features_for_one(self, x):
		NN_output = self.NN.kneighbors(x)

		neighs      = NN_output[1][0]
		neighs_dist = NN_output[0][0]

		neighs_y = self.y_train[neighs]

		return_list = []

		"""
		1. Fraction of objects of every class
		"""

		for k in self.k_list:
			NN_output = self.NN.kneighbors(x, n_neighbors=k)

			neighs      = NN_output[1][0]
			neighs_dist = NN_output[0][0]

			neighs_y        = self.y_train[neighs]
			
			neighs_bincount = np.bincount(neighs_y, minlength=3) # for binary classification
			neighs_bincount = neighs_bincount / neighs_bincount.sum()
				
			return_list += [neighs_bincount]

		"""
		2. Same label streak: largest number N, such that N nearest neighbors have same label
		"""

		largest_streak = np.where(neighs_y[1:] != neighs_y[:-1])

		if len(largest_streak[0]) == 0:
			largest_streak = len(neighs_y)
		else:
			largest_streak = largest_streak[0][0] + 1 

		feats = [largest_streak]

		return_list += [feats]


		"""
		3. Minimum distance of objects of each class
		"""

		feats = []

		for c in range(2):
			c_indices = np.where(neighs_y == c)[0]

			if len(c_indices) == 0:
				feats.append(999.0)
			else:
				feats.append(neighs_dist[c_indices[0]])

		return_list += [feats]

		"""
		4. Minimum normalized distance to objects of each class
		   as 3. but normalized.
		"""	

		feats = []

		for c in range(2):
			c_indices = np.where(neighs_y == c)[0]

			if len(c_indices) == 0:
				feats.append(999.0)
			else:
				feats.append(neighs_dist[c_indices[0]] / (neighs_dist[0] + self.eps))

		return_list += [feats]

		"""
		5. a) Distance to the k-th neighbor
		   b) Distance to kth neighbor distance to first neighbor
		"""

		for k in self.k_list:
			NN_output   = self.NN.kneighbors(x, n_neighbors=k)
			neighs      = NN_output[1][0]
			neighs_dist = NN_output[0][0]
			
			feat_51 = neighs_dist[-1]
			feat_52 = neighs_dist[-1] / (neighs_dist[0] + self.eps)
			
			return_list += [[feat_51, feat_52]]

		'''
			6. Mean distance to neighbors of each class for each K from `k_list` 
				   For each class select the neighbors of that class among K nearest neighbors 
				   and compute the average distance to those objects
				   
				   If there are no objects of a certain class among K neighbors, set mean distance to 999
				   
		'''
		for k in self.k_list:
			
			NN_output   = self.NN.kneighbors(x, n_neighbors=k)
			neighs      = NN_output[1][0]
			neighs_dist = NN_output[0][0]
			
			neighs_y    = self.y_train[neighs] 
			counts      = np.bincount(neighs_y, minlength=3)
			feats       = np.bincount(neighs_y, neighs_dist, minlength=3)
			feats       = feats / counts
			feats[~np.isfinite(feats)] = 999.0
			
			return_list += [feats]


		knn_feats = np.hstack(return_list)
			
		return knn_feats