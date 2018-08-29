import pandas as pd
import numpy as np
import gc
import feather

from sklearn.base import BaseEstimator, ClassifierMixin

import xgboost as xgb

SEED = 1231
np.random.seed(SEED)

class XGBoostLeaves(BaseEstimator, ClassifierMixin):
	params = {
		'objective': 'binary:logistic',
		'eta': .1,
		'max_depth': 6,
		'silent': 1
	}

	def __init__(self, num_leaves):
		self.num_leaves = num_leaves
		
	def fit(self, X, y=None):
		dtrain = xgb.DMatrix(X, y)
		self.model = xgb.train(self.params, dtrain, self.num_leaves)

	def predict(self, X):
		return self.model.predict(xgb.DMatrix(X), pred_leaf=True)
		