import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

SEED = 1231
np.random.seed(SEED)

#TODO: Think what all methods can be moved to base class which would be beneficial for
# other experiments.

class BaseModel:
	def __init__(self):
		pass

	def reduce_mem_usage(self, df):
		"""	
		iterate through all the columns of a dataframe and modify the data type
		to reduce memory usage.        
		"""
		start_mem = df.memory_usage().sum() / 1024**2
		print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
		
		for col in df.columns:
			col_type = df[col].dtype
			
			if col_type != object:
				c_min = df[col].min()
				c_max = df[col].max()
				if str(col_type)[:3] == 'int':
					if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
						df[col] = df[col].astype(np.int8)
					elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
						df[col] = df[col].astype(np.int16)
					elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
						df[col] = df[col].astype(np.int32)
					elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
						df[col] = df[col].astype(np.int64)  
				else:
					if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
						df[col] = df[col].astype(np.float16)
					elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
						df[col] = df[col].astype(np.float32)
					else:
						df[col] = df[col].astype(np.float64)
			else:
				df[col] = df[col].astype('category')

		end_mem = df.memory_usage().sum() / 1024**2
		print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
		print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
		
		return df

	def create_fold(self, data, seed):
		dtr, dte, _, _ = train_test_split(data, 
											data.TARGET, 
											stratify=data.TARGET, 
											test_size=.3,
											random_state=seed
											)
		return dtr, dte

	def train_lgb(self, X, y, Xte, yte, **params):
		num_boost_round = params['num_boost_round']
		del params['num_boost_round']

		ltrain = lgb.Dataset(X, y, 
							feature_name=X.columns.tolist())
		
		m = None

		if len(yte):
			lval = lgb.Dataset(Xte, yte, feature_name=X.columns.tolist())
			
			valid_sets  = [ltrain, lval]
			valid_names = ['train', 'val']

			early_stopping_rounds = 100

			m = lgb.train(params, 
						ltrain, 
						num_boost_round, 
						valid_sets=valid_sets, 
						valid_names=valid_names, 
						early_stopping_rounds=early_stopping_rounds, 
						verbose_eval=20)
			
			# feature importances
			feature_names = m.feature_name()
			feature_imp   = m.feature_importance()

			print(pd.DataFrame({'features': feature_names,
								'imp': feature_imp
								}).sort_values(by='imp', ascending=False))
		
		else:
			m = lgb.train(params,
						  ltrain,
						  num_boost_round
						)
		
		return m


	def evaluate_lgb(self, Xte, yte, model):
		yhat  = None
		score = None

		if len(yte):
			print('Best Iteration: {}'.format(model.best_iteration))

			yhat  = model.predict(Xte, num_iteration=model.best_iteration)
			score = roc_auc_score(yte, yhat)
			print('AUC: {}'.format(score))
		else:
			yhat = model.predict(Xte)
		
		return yhat, score