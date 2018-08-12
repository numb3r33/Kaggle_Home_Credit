import pandas as pd
import numpy as np

import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import BayesianOptimization
from bayes_opt import BayesianOptimization

import time

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

    def fill_infrequent_values(self, data):
        # replace feature values with frequency less 20 with -100
        for f in data.select_dtypes(include=['int8']).columns:
            if data[f].nunique() > 2:        
                low_freq_values = data[f].value_counts()
                low_freq_values = low_freq_values[low_freq_values < 20].index.values
                
                if len(low_freq_values) > 0:
                    print('Feature: {}'.format(f))
                    data.loc[data[f].isin(low_freq_values), f] = -100
        
        return data

    def prepare_ohe(self, data, cols, drop_col=False):
        for col in cols:
            ohe_df = pd.get_dummies(data[col].astype(np.str), dummy_na=True, prefix=f'{col}_')

            # drop the column passed
            if drop_col:
                data.drop(col, axis=1, inplace=True)
            
            data = pd.concat((data, ohe_df), axis=1)

        return data

    def create_fold(self, data, seed):
        dtr, dte, _, _ = train_test_split(data, 
                                            data.TARGET, 
                                            stratify=data.TARGET, 
                                            test_size=.3,
                                            random_state=seed
                                            )
        return dtr, dte

    def get_sample(self, train, sample_size, seed):
        _, train, _, _ = train_test_split(train, 
                                          train.TARGET, 
                                          stratify=train.TARGET,
                                          test_size=sample_size,
                                          random_state=seed
                                          )
        return train

    def train_lgb(self, X, y, Xte, yte, categorical_feature='auto', **params):
        print()
        print('Train LightGBM classifier ...')
        print('*' * 100)
        print()

        num_boost_round       = params['num_boost_round']
        early_stopping_rounds = params['early_stopping_rounds']

        del params['num_boost_round'], params['early_stopping_rounds']

        ltrain = lgb.Dataset(X, y, 
                            feature_name=X.columns.tolist(),
                            categorical_feature=categorical_feature
                            )
        
        m       = None
        feat_df = None

        # start time counter
        t0 = time.time()

        if len(yte):
            lval = lgb.Dataset(Xte, yte, feature_name=X.columns.tolist())
            
            valid_sets  = [ltrain, lval]
            valid_names = ['train', 'val']

            m = lgb.train(params, 
                        ltrain, 
                        num_boost_round, 
                        valid_sets=valid_sets, 
                        valid_names=valid_names, 
                        early_stopping_rounds=early_stopping_rounds, 
                        verbose_eval=20)
            
            # feature importances
            feature_names = m.feature_name()
            feature_imp   = m.feature_importance(importance_type='gain')

            feat_df = pd.DataFrame({'features': feature_names,
                                    'imp': feature_imp
                                   }).sort_values(by='imp', ascending=False)
        
        else:
            m = lgb.train(params,
                          ltrain,
                          num_boost_round
                        )

            # feature importances
            feature_names = m.feature_name()
            feature_imp   = m.feature_importance(importance_type='gain')

            feat_df = pd.DataFrame({'features': feature_names,
                                    'imp': feature_imp
                                   }).sort_values(by='imp', ascending=False)

        print('Took: {} seconds to generate...'.format(time.time() - t0))
        
        return m, feat_df
    
    def get_folds(self, X, cv_df):
        FOLD_NUM = [0, 2, 3, 4, 5]
        
        for fold in FOLD_NUM:
            test_idx  = list(cv_df[f'F{fold}'].values)
            train_idx = list(set(X.index) - set(test_idx))

            yield train_idx, test_idx
    
    def cross_validate(self, Xtr, ytr, params, cv_adversarial_filepath=None, categorical_feature='auto'):
        num_boost_round       = params['num_boost_round']
        early_stopping_rounds = params['early_stopping_rounds']

        del params['num_boost_round'], params['early_stopping_rounds']

        # start time counter
        t0     = time.time()
        
        ltrain = lgb.Dataset(Xtr, ytr, feature_name=Xtr.columns.tolist(), categorical_feature=categorical_feature)

        if cv_adversarial_filepath is not None:
            cv_df = pd.read_csv(cv_adversarial_filepath)

            cv     = lgb.cv(params, 
                            ltrain,
                            folds=self.get_folds(Xtr, cv_df), 
                            num_boost_round=num_boost_round, 
                            early_stopping_rounds=early_stopping_rounds,
                            seed=params['seed'],
                            verbose_eval=20
                        )
        else:
            cv     = lgb.cv(params, 
                        ltrain,
                        num_boost_round=num_boost_round, 
                        early_stopping_rounds=early_stopping_rounds,
                        seed=params['seed'],
                        verbose_eval=20
                    )

        print('\nTook: {} seconds'.format(time.time() - t0))
        
        return pd.DataFrame(cv)

    def optimize_lgb(self, Xtr, ytr, Xte, yte, param_grid):
        
        params = {
            'feature_fraction_seed': 4457,
            'bagging_seed': 4457,
            'nthread': 8,
            'verbose': -1,
            'seed': 4457,
            'num_boost_round': 20000,
            'early_stopping_rounds': 100
        }

        def fun(**kw):
            params = {}

            for k in kw:
                if type(param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]

            print('Trying {} .....'.format(params))

            model, _ = self.train_lgb(Xtr, ytr, Xte, yte, **params)

            print('Score: {} at iteration: {}'%(model.best_score, model.best_iteration))
            return model.best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=2)

        best_score  = opt.res['max']['max_val']
        best_params = opt.res['max']['max_params']

        print('Best AUC score: {}, params: {}'.format(best_score, best_params))

        return best_score, best_params


    def cross_validate_xgb(self, Xtr, ytr, params, cv_adversarial_filepath=None):
        num_boost_round       = params['num_boost_round']
        early_stopping_rounds = params['early_stopping_rounds']

        del params['num_boost_round'], params['early_stopping_rounds']

        # start time counter
        t0     = time.time()
        dtrain = xgb.DMatrix(Xtr, ytr, feature_names=Xtr.columns.tolist())

        if cv_adversarial_filepath is not None:
            cv_df = pd.read_csv(cv_adversarial_filepath)

            cv     = xgb.cv(params, 
                            dtrain,
                            folds=self.get_folds(Xtr, cv_df), 
                            num_boost_round=num_boost_round, 
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=20
                        )
        else:
            cv     = xgb.cv(params, 
                        dtrain,
                        num_boost_round=num_boost_round, 
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=20
                    )

        print('\nTook: {} seconds'.format(time.time() - t0))
        
        return pd.DataFrame(cv)

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

    def train_xgb(self, X, y, Xte, yte, **params):
        print()
        print('Training XGBOOST Classifier ....')
        print('*' * 80)

        num_boost_round = params['num_boost_round']
        del params['num_boost_round']

        dtrain = xgb.DMatrix(X, y, 
                            feature_names=X.columns.tolist())
        
        m       = None
        feat_df = None

        # start time counter
        t0 = time.time()

        if len(yte):
            dval = xgb.DMatrix(Xte, yte, feature_names=X.columns.tolist())
            
            watchlist   = [(dtrain, 'train'), (dval, 'val')]

            m = xgb.train(params, 
                          dtrain, 
                          num_boost_round, 
                          evals=watchlist, 
                          early_stopping_rounds=early_stopping_rounds, 
                          verbose_eval=100)
            
            # feature importances
            feature_imp   = m.get_score()
            feat_df       = pd.DataFrame({'features': list(feature_imp.keys()),
                                    'imp': list(feature_imp.values())
                                   }).sort_values(by='imp', ascending=False)
        
        else:
            m = xgb.train(params,
                          dtrain,
                          num_boost_round
                        )

            # feature importances
            feature_imp   = m.get_score()

            feat_df = pd.DataFrame({'features': list(feature_imp.keys()),
                                    'imp': list(feature_imp.values())
                                   }).sort_values(by='imp', ascending=False)
        
        print('Took: {} seconds to generate...'.format(time.time() - t0))
        
        return m, feat_df

    def evaluate_xgb(self, Xte, yte, model):
        yhat  = None
        score = None

        dval  = xgb.DMatrix(Xte, feature_names=Xte.columns.tolist())
 
        if len(yte):
            print('Best Iteration: {}'.format(model.best_iteration))
            yhat  = model.predict(dval, ntree_limit=model.best_iteration)
            
            score = roc_auc_score(yte, yhat)
            print('AUC: {}'.format(score))
        else:
            yhat = model.predict(dval)
        
        return yhat, score

    def add_pca_components(self, data, PCA_PARAMS):
        # preprocess for pca
        SKIP_COLS = ['SK_ID_CURR', 'TARGET']
        data = data.loc[:, data.columns.drop(SKIP_COLS)]

        for col in data.columns:
            # replace inf with np.nan
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            
            # fill missing values with median
            if data[col].isnull().sum():
                if pd.isnull(data[col].median()):
                    data[col] = data[col].fillna(-1)
                else:
                    data[col] = data[col].fillna(data[col].median())

        pca  = self.fit_pca(data, **PCA_PARAMS)
        data = self.transform_pca(data)

        data = pd.DataFrame(data, columns=[f'pca_{i}' for i in range(PCA_PARAMS['n_components'])])
        
        return data

    def oof_preds(self, X, y, Xte, model):
        oof_preds = cross_val_predict(model, X.values, y.values, cv=5, method='predict_proba')

        model.fit(X.values, y.values)
        test_preds = model.predict_proba(Xte)[:, 1]

        return oof_preds[:, 1], test_preds
    
    def fit_pca(self, X, **pca_params):
        scaler = StandardScaler()
        X      = scaler.fit_transform(X)

        pca = PCA(**pca_params)
        pca.fit(X)

        self.scaler = scaler
        self.pca    = pca

        return pca

    def transform_pca(self, X):
        X = self.scaler.transform(X)
        return self.pca.transform(X)    