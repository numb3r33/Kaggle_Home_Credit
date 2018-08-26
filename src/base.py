import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import gc

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool, cv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold

from bayes_opt import BayesianOptimization
from MulticoreTSNE import MulticoreTSNE as TSNE

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
        FOLD_NUM = [0, 1, 2, 3, 4]
        
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
                            verbose_eval=100
                        )
        else:
            cv     = lgb.cv(params, 
                        ltrain,
                        num_boost_round=num_boost_round, 
                        early_stopping_rounds=early_stopping_rounds,
                        seed=params['seed'],
                        verbose_eval=100
                    )

        print('\nTook: {} seconds'.format(time.time() - t0))
        
        return pd.DataFrame(cv)
    
    def cv_predict(self, train, test, feature_list, params, cv_adversarial_filepath, categorical_feature='auto'):
        num_boost_round       = params['num_boost_round']
        early_stopping_rounds = params['early_stopping_rounds']

        del params['num_boost_round'], params['early_stopping_rounds']

        # start time counter
        test_preds = np.zeros(shape=(len(test))) 
        hold_auc   = []
        fold_trees = []

        for fold in ['F0', 'F1', 'F2', 'F3', 'F4']:
            print('Fold: {}'.format(fold))
            
            # train with a different seed
            params['seed'] += 100

            print('Seed : {}'.format(params['seed']))

            # load test fold indicators
            ite  = pd.read_csv(cv_adversarial_filepath, usecols=[fold])[fold].values
            itr  = np.array(list(set(train.index) - set(ite)))

            tr  = train.loc[train.index.isin(itr)]
            te  = train.loc[train.index.isin(ite)]

            Xtr = tr.loc[:, feature_list]
            ytr = tr.loc[:, 'TARGET']

            Xte = te.loc[:, feature_list]
            yte = te.loc[:, 'TARGET']

            ltrain = lgb.Dataset(Xtr, ytr, feature_name=Xtr.columns.tolist(), categorical_feature=categorical_feature)
            leval  = lgb.Dataset(Xte, yte, feature_name=Xte.columns.tolist(), categorical_feature=categorical_feature)

            valid_sets  = [ltrain, leval]
            valid_names = ['train', 'eval']

            model  = lgb.train(params, 
                               ltrain, 
                               num_boost_round, 
                               valid_sets=valid_sets, 
                               valid_names=valid_names, 
                               early_stopping_rounds=early_stopping_rounds, 
                               verbose_eval=20
                               )

            hold_preds = model.predict(Xte, num_iteration=model.best_iteration)
            test_preds += (model.predict(test.loc[:, feature_list], num_iteration=model.best_iteration) * (1 / 5))
            fold_auc = roc_auc_score(yte, hold_preds)
            fold_trees.append(model.best_iteration)

            print('Best iteration: {}'.format(model.best_iteration))
            print('AUC on holdout set: {}'.format(fold_auc))

            hold_auc.append(fold_auc)

        return np.array(hold_auc), test_preds, fold_trees

    def predict_test(self, train, test, feature_list, params, save_path, 
                     kfold_seeds=[2017, 2016, 2015, 2014, 2013], 
                     n_folds=5, 
                     categorical_feature='auto'):
        
        num_boost_round       = params['num_boost_round']
        early_stopping_rounds = params['early_stopping_rounds']

        del params['num_boost_round'], params['early_stopping_rounds']

        pred_valid = np.zeros((train.shape[0], len(kfold_seeds)))
        pred_test  = np.zeros((test.shape[0], len(kfold_seeds)))

        X = train.loc[:, feature_list]
        y = train.loc[:, 'TARGET']

        X_test = test.loc[:, feature_list] 

        del train, test
        gc.collect()

        t0 = time.time()

        # train
        for bag_idx, kfold_seed in enumerate(kfold_seeds):
            kf = KFold(n_folds, shuffle=True, random_state=kfold_seed)

            for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                lg_train = lgb.Dataset(X_train, y_train)
                lg_valid = lgb.Dataset(X_valid, y_valid)

                evals_result = {}
                model = lgb.train(params, 
                                  lg_train, 
                                  num_boost_round, 
                                  valid_sets=[lg_valid], 
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose_eval=100,
                                  evals_result=evals_result
                                  )

                fig, ax = plt.subplots(figsize=(12, 18))
                lgb.plot_importance(model, max_num_features=50, importance_type='gain', height=0.8, ax=ax)
                ax.grid(False)

                plt.title('Light GBM - Feature Importance', fontsize=15)
                plt.savefig(save_path + f'importance_{bag_idx}_{fold_idx}.png')
                plt.close()


                pred_valid[valid_idx, bag_idx] = model.predict(X_valid, num_iteration=model.best_iteration)
                auc = roc_auc_score(y_valid, pred_valid[valid_idx, bag_idx])

                print('{}-fold auc: {}'.format(fold_idx, auc))
                pred_test[:, bag_idx] += model.predict(X_test, num_iteration=model.best_iteration) / len(kfold_seeds)

            auc = roc_auc_score(y, pred_valid[:, bag_idx])
            print('{}-bag auc: {}'.format(bag_idx, auc))
        
        print('Took: {} seconds to prepare oof and test predictions'.format(time.time() - t0))

        auc = roc_auc_score(y, pred_valid.mean(axis=1))
        print('total auc: {}'.format(auc))

        pred_test_final = pred_test.mean(axis=1)

        return auc, pred_valid, pred_test, pred_test_final   

    def predict_test_xgb(self, 
                         train, 
                         test, 
                         feature_list, 
                         params, 
                         save_path, 
                         kfold_seeds = [2017, 2016, 2015, 2014, 2013], 
                         n_folds=5, 
                         categorical_feature='auto'):
        
        num_boost_round       = params['num_boost_round']
        early_stopping_rounds = params['early_stopping_rounds']

        del params['num_boost_round'], params['early_stopping_rounds']

        pred_valid = np.zeros((train.shape[0], len(kfold_seeds)))
        pred_test  = np.zeros((test.shape[0], len(kfold_seeds)))

        X = train.loc[:, feature_list]
        y = train.loc[:, 'TARGET']

        X_test   = test.loc[:, feature_list] 
        xgb_test = xgb.DMatrix(X_test)

        del train, test
        gc.collect()

        t0 = time.time()

        # train
        for bag_idx, kfold_seed in enumerate(kfold_seeds):
            kf = KFold(n_folds, shuffle=True, random_state=kfold_seed)

            for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                xgb_train = xgb.DMatrix(X_train, y_train)
                xgb_valid = xgb.DMatrix(X_valid, y_valid)

                evals_result = {}
                model = xgb.train(params, 
                                  xgb_train, 
                                  num_boost_round, 
                                  evals=[(xgb_valid, 'valid')], 
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose_eval=100,
                                  evals_result=evals_result
                                  )

                fig, ax = plt.subplots(figsize=(12, 18))
                xgb.plot_importance(model, max_num_features=50, importance_type='gain', height=0.8, ax=ax)
                ax.grid(False)

                plt.title('XGBoost - Feature Importance', fontsize=15)
                plt.savefig(save_path + f'importance_{bag_idx}_{fold_idx}.png')
                plt.close()


                pred_valid[valid_idx, bag_idx] = model.predict(xgb_valid, ntree_limit=model.best_iteration)
                auc = roc_auc_score(y_valid, pred_valid[valid_idx, bag_idx])

                print('{}-fold auc: {}'.format(fold_idx, auc))
                pred_test[:, bag_idx] += model.predict(xgb_test, ntree_limit=model.best_iteration) / len(kfold_seeds)

            auc = roc_auc_score(y, pred_valid[:, bag_idx])
            print('{}-bag auc: {}'.format(bag_idx, auc))
        
        print('Took: {} seconds to prepare oof and test predictions'.format(time.time() - t0))

        auc = roc_auc_score(y, pred_valid.mean(axis=1))
        print('total auc: {}'.format(auc))

        pred_test_final = pred_test.mean(axis=1)

        return auc, pred_valid, pred_test, pred_test_final           
                
    
    def predict_test_cb(self, train, test, feature_list, params, n_folds=5, categorical_feature='auto'):
        kfold_seeds = [2017, 2016, 2015, 2014, 2013]

        pred_valid = np.zeros((train.shape[0], len(kfold_seeds)))
        pred_test  = np.zeros((test.shape[0], len(kfold_seeds)))

        X = train.loc[:, feature_list]
        y = train.loc[:, 'TARGET']

        X_test   = test.loc[:, feature_list] 
        
        del train, test
        gc.collect()

        t0 = time.time()

        params.update({
            'use_best_model': True
        })

        # train
        for bag_idx, kfold_seed in enumerate(kfold_seeds):
            kf = KFold(n_folds, shuffle=True, random_state=kfold_seed)

            for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = CatBoostClassifier(**params)
                model.fit(X_train, y_train)

                pred_valid[valid_idx, bag_idx] = model.predict_proba(X_valid)[:, 1]
                auc = roc_auc_score(y_valid, pred_valid[valid_idx, bag_idx])

                print('{}-fold auc: {}'.format(fold_idx, auc))
                pred_test[:, bag_idx] += model.predict_proba(X_test)[:, 1] / len(kfold_seeds)

            auc = roc_auc_score(y, pred_valid[:, bag_idx])
            print('{}-bag auc: {}'.format(bag_idx, auc))
        
        print('Took: {} seconds to prepare oof and test predictions'.format(time.time() - t0))

        auc = roc_auc_score(y, pred_valid.mean(axis=1))
        print('total auc: {}'.format(auc))

        pred_test_final = pred_test.mean(axis=1)

        return auc, pred_valid, pred_test, pred_test_final

    def predict_test_rf(self, train, test, feature_list, params, n_folds=5, categorical_feature='auto'):
        kfold_seeds = [2017, 2016, 2015, 2014, 2013]

        pred_valid = np.zeros((train.shape[0], len(kfold_seeds)))
        pred_test  = np.zeros((test.shape[0], len(kfold_seeds)))

        X = train.loc[:, feature_list]
        y = train.loc[:, 'TARGET']

        X_test   = test.loc[:, feature_list]

        data = pd.concat((X, X_test))

        # preprocess for RF
        for col in data.columns:
            # replace inf with np.nan
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            
            # fill missing values with median
            if data[col].isnull().sum():
                if pd.isnull(data[col].median()):
                    data[col] = data[col].fillna(-1)
                else:
                    data[col] = data[col].fillna(data[col].median())
        
        X = data.iloc[:len(X)]
        X_test = data.iloc[len(X):]

        del train, test, data
        gc.collect()

        t0 = time.time()

        # train
        for bag_idx, kfold_seed in enumerate(kfold_seeds):
            kf = KFold(n_folds, shuffle=True, random_state=kfold_seed)

            for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)

                pred_valid[valid_idx, bag_idx] = model.predict_proba(X_valid)[:, 1]
                auc = roc_auc_score(y_valid, pred_valid[valid_idx, bag_idx])

                print('{}-fold auc: {}'.format(fold_idx, auc))
                pred_test[:, bag_idx] += model.predict_proba(X_test)[:, 1] / len(kfold_seeds)

            auc = roc_auc_score(y, pred_valid[:, bag_idx])
            print('{}-bag auc: {}'.format(bag_idx, auc))
        
        print('Took: {} seconds to prepare oof and test predictions'.format(time.time() - t0))

        auc = roc_auc_score(y, pred_valid.mean(axis=1))
        print('total auc: {}'.format(auc))

        pred_test_final = pred_test.mean(axis=1)

        return auc, pred_valid, pred_test, pred_test_final

    def predict_test_etc(self, train, test, feature_list, params, n_folds=5, categorical_feature='auto'):
        kfold_seeds = [2017, 2016, 2015, 2014, 2013]

        pred_valid = np.zeros((train.shape[0], len(kfold_seeds)))
        pred_test  = np.zeros((test.shape[0], len(kfold_seeds)))

        X = train.loc[:, feature_list]
        y = train.loc[:, 'TARGET']

        X_test   = test.loc[:, feature_list]

        data = pd.concat((X, X_test))

        # preprocess for RF
        for col in data.columns:
            # replace inf with np.nan
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            
            # fill missing values with median
            if data[col].isnull().sum():
                if pd.isnull(data[col].median()):
                    data[col] = data[col].fillna(-1)
                else:
                    data[col] = data[col].fillna(data[col].median())
        
        X = data.iloc[:len(X)]
        X_test = data.iloc[len(X):]

        del train, test, data
        gc.collect()

        t0 = time.time()

        # train
        for bag_idx, kfold_seed in enumerate(kfold_seeds):
            kf = KFold(n_folds, shuffle=True, random_state=kfold_seed)

            for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = ExtraTreesClassifier(**params)
                model.fit(X_train, y_train)

                pred_valid[valid_idx, bag_idx] = model.predict_proba(X_valid)[:, 1]
                auc = roc_auc_score(y_valid, pred_valid[valid_idx, bag_idx])

                print('{}-fold auc: {}'.format(fold_idx, auc))
                pred_test[:, bag_idx] += model.predict_proba(X_test)[:, 1] / len(kfold_seeds)

            auc = roc_auc_score(y, pred_valid[:, bag_idx])
            print('{}-bag auc: {}'.format(bag_idx, auc))
        
        print('Took: {} seconds to prepare oof and test predictions'.format(time.time() - t0))

        auc = roc_auc_score(y, pred_valid.mean(axis=1))
        print('total auc: {}'.format(auc))

        pred_test_final = pred_test.mean(axis=1)

        return auc, pred_valid, pred_test, pred_test_final
    
    def predict_test_log(self, 
                         train, 
                         test, 
                         feature_list, 
                         params, 
                         kfold_seeds=[2017, 2016, 2015, 2014, 2013], 
                         n_folds=5, 
                         categorical_feature='auto'):
        
        pred_valid = np.zeros((train.shape[0], len(kfold_seeds)))
        pred_test  = np.zeros((test.shape[0], len(kfold_seeds)))

        X = train.loc[:, feature_list]
        y = train.loc[:, 'TARGET']

        X_test   = test.loc[:, feature_list]

        data = pd.concat((X, X_test))

        # preprocess for RF
        for col in data.columns:
            # replace inf with np.nan
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            
            # fill missing values with median
            if data[col].isnull().sum():
                if pd.isnull(data[col].median()):
                    data[col] = data[col].fillna(-1)
                else:
                    data[col] = data[col].fillna(data[col].median())
        
        X = data.iloc[:len(X)]
        X_test = data.iloc[len(X):]

        del train, test, data
        gc.collect()

        t0 = time.time()

        # train
        for bag_idx, kfold_seed in enumerate(kfold_seeds):
            kf = KFold(n_folds, shuffle=True, random_state=kfold_seed)

            for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = ExtraTreesClassifier(**params)
                model.fit(X_train, y_train)

                pred_valid[valid_idx, bag_idx] = model.predict_proba(X_valid)[:, 1]
                auc = roc_auc_score(y_valid, pred_valid[valid_idx, bag_idx])

                print('{}-fold auc: {}'.format(fold_idx, auc))
                pred_test[:, bag_idx] += model.predict_proba(X_test)[:, 1] / len(kfold_seeds)

            auc = roc_auc_score(y, pred_valid[:, bag_idx])
            print('{}-bag auc: {}'.format(bag_idx, auc))
        
        print('Took: {} seconds to prepare oof and test predictions'.format(time.time() - t0))

        auc = roc_auc_score(y, pred_valid.mean(axis=1))
        print('total auc: {}'.format(auc))

        pred_test_final = pred_test.mean(axis=1)

        return auc, pred_valid, pred_test, pred_test_final


    def optimize_lgb(self, Xtr, ytr, Xte, yte, param_grid):
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.03,
            'feature_fraction_seed': 4457,
            'bagging_seed': 4457,
            'nthread': 8,
            'verbose': -1,
            'seed': 4457,
            'num_boost_round': 20000,
            'early_stopping_rounds': 100
        }

        def fun(**kw):

            for k in kw:
                if type(param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]

            print('Trying {} .....'.format(params))

            model, _ = self.train_lgb(Xtr, ytr, Xte, yte, **params)

            print('Score: {} at iteration: {}'.format(model.best_score, model.best_iteration))
            return model.best_score['val']['auc']

        opt = BayesianOptimization(fun, param_grid, random_state=4457)
        opt.maximize(n_iter=15)

        print('Optimization object: {}'.format(opt.res['max']))
        
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
                            verbose_eval=100
                        )
        else:
            cv     = xgb.cv(params, 
                        dtrain,
                        num_boost_round=num_boost_round, 
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=100
                    )

        print('\nTook: {} seconds'.format(time.time() - t0))
        
        return pd.DataFrame(cv)

    def cross_validate_rf(self, Xtr, ytr, params, cv_adversarial_filepath):
        # start time counter
        t0     = time.time()
        
        FOLD_NUM = [0, 1, 2, 3, 4]
        
        # load cross validation indices file
        cv_df = pd.read_csv(cv_adversarial_filepath)

        model = RandomForestClassifier(**params)

        # preprocess for RF
        for col in Xtr.columns:
            # replace inf with np.nan
            Xtr[col] = Xtr[col].replace([np.inf, -np.inf], np.nan)
            
            # fill missing values with median
            if Xtr[col].isnull().sum():
                if pd.isnull(Xtr[col].median()):
                    Xtr[col] = Xtr[col].fillna(-1)
                else:
                    Xtr[col] = Xtr[col].fillna(Xtr[col].median())

        auc = []

        for fold in FOLD_NUM:
            test_idx  = list(cv_df[f'F{fold}'].values)
            train_idx = list(set(Xtr.index) - set(test_idx))

            x_trn = Xtr.iloc[train_idx]
            y_trn = ytr.iloc[train_idx]

            x_val = Xtr.iloc[test_idx]
            y_val = ytr.iloc[test_idx]

            model.fit(x_trn, y_trn)
            fold_preds = model.predict_proba(x_val)[:, 1]

            fold_auc = roc_auc_score(y_val, fold_preds)

            auc.append(fold_auc)
        
        auc = np.array(auc)

        return np.mean(auc), np.std(auc)

        def cross_validate_log(self, Xtr, ytr, params, cv_adversarial_filepath):
            # start time counter
            t0     = time.time()
            
            FOLD_NUM = [0, 1, 2, 3, 4]
            
            # load cross validation indices file
            cv_df = pd.read_csv(cv_adversarial_filepath)

            model = LogisticRegression(**params)

            # preprocess for RF
            for col in Xtr.columns:
                # replace inf with np.nan
                Xtr[col] = Xtr[col].replace([np.inf, -np.inf], np.nan)
                
                # fill missing values with median
                if Xtr[col].isnull().sum():
                    if pd.isnull(Xtr[col].median()):
                        Xtr[col] = Xtr[col].fillna(-1)
                    else:
                        Xtr[col] = Xtr[col].fillna(Xtr[col].median())

            auc = []

            for fold in FOLD_NUM:
                test_idx  = list(cv_df[f'F{fold}'].values)
                train_idx = list(set(Xtr.index) - set(test_idx))

                x_trn = Xtr.iloc[train_idx]
                y_trn = ytr.iloc[train_idx]

                x_val = Xtr.iloc[test_idx]
                y_val = ytr.iloc[test_idx]

                model.fit(x_trn, y_trn)
                fold_preds = model.predict_proba(x_val)[:, 1]

                fold_auc = roc_auc_score(y_val, fold_preds)

                auc.append(fold_auc)
            
            auc = np.array(auc)

            return np.mean(auc), np.std(auc)


    def cross_validate_etc(self, Xtr, ytr, params, cv_adversarial_filepath):
        # start time counter
        t0     = time.time()
        
        FOLD_NUM = [0, 1, 2, 3, 4]
        
        # load cross validation indices file
        cv_df = pd.read_csv(cv_adversarial_filepath)

        model = ExtraTreesClassifier(**params)

        # preprocess for RF
        for col in Xtr.columns:
            # replace inf with np.nan
            Xtr[col] = Xtr[col].replace([np.inf, -np.inf], np.nan)
            
            # fill missing values with median
            if Xtr[col].isnull().sum():
                if pd.isnull(Xtr[col].median()):
                    Xtr[col] = Xtr[col].fillna(-1)
                else:
                    Xtr[col] = Xtr[col].fillna(Xtr[col].median())

        auc = []

        for fold in FOLD_NUM:
            test_idx  = list(cv_df[f'F{fold}'].values)
            train_idx = list(set(Xtr.index) - set(test_idx))

            x_trn = Xtr.iloc[train_idx]
            y_trn = ytr.iloc[train_idx]

            x_val = Xtr.iloc[test_idx]
            y_val = ytr.iloc[test_idx]

            model.fit(x_trn, y_trn)
            fold_preds = model.predict_proba(x_val)[:, 1]

            fold_auc = roc_auc_score(y_val, fold_preds)

            auc.append(fold_auc)
        
        auc = np.array(auc)

        return np.mean(auc), np.std(auc)


    
    def cross_validate_cb(self, Xtr, ytr, params, cv_adversarial_filepath):
        # start time counter
        t0     = time.time()
        
        FOLD_NUM = [0, 1, 2, 3, 4]
        
        # load cross validation indices file
        cv_df = pd.read_csv(cv_adversarial_filepath)

        model = CatBoostClassifier(**params)

        auc = []

        for fold in FOLD_NUM:
            test_idx  = list(cv_df[f'F{fold}'].values)
            train_idx = list(set(Xtr.index) - set(test_idx))

            x_trn = Xtr.iloc[train_idx]
            y_trn = ytr.iloc[train_idx]

            x_val = Xtr.iloc[test_idx]
            y_val = ytr.iloc[test_idx]

            # train CatBoost Classifier
            model.fit(x_trn, y_trn, verbose=False)
            
            fold_preds = model.predict_proba(x_val)[:, 1]
            fold_auc   = roc_auc_score(y_val, fold_preds)

            auc.append(fold_auc)
        
        auc = np.array(auc)

        return np.mean(auc), np.std(auc)


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

    def add_tsne_components(self, data):
        t0 = time.time()

        # preprocess for tsne
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

        data  = self.fit_transform_tsne(data)
        data = pd.DataFrame(data, columns=[f'tsne_{i}' for i in range(2)])
        
        print('Took: {} seconds to generate T-SNE embeddings'.format(time.time() - t0))

        return data


    def cross_validate_sklearn(self, X, y, model, seed):
        print('Cross validating {}'.format(model))

        t0 = time.time()

        skf       = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

        print('Took: {} seconds'.format(time.time() - t0))

        return cv_scores


    def train_sklearn(self, X, y, Xte, yte, model):
        print('Training a {} Classifier ...'.format(model))

        t0 = time.time()
        
        if len(yte):
            model.fit(X, y)
            print('Took: {} seconds'.format((time.time() - t0)))
            
            hold_preds = model.predict_proba(Xte)[:, 1]
            print('Holdout Score: {}'.format(roc_auc_score(yte, hold_preds)))
            
            return model, hold_preds 
        else:
            print('Full Training ...')
            model.fit(X, y)
            return model, model.predict_proba(Xte)[:, 1]

    def evaluate_sklearn(self, Xte, yte, model):
        yhat  = None
        score = None

        if len(yte):
            yhat  = model.predict_proba(Xte)[:, 1]
            
            score = roc_auc_score(yte, yhat)
            print('AUC: {}'.format(score))
        else:
            yhat = model.predict_proba(Xte)[:, 1]
        
        return yhat, score


    def get_oof_preds(self, X, y, Xte, model):
        t0 = time.time()

        oof_preds = cross_val_predict(model, X.values, y.values, cv=5, method='predict_proba')

        print('Took: {} seconds to generate oof preds'.format(time.time() - t0))

        model.fit(X.values, y.values)
        test_preds = model.predict_proba(Xte.values)[:, 1]

        return oof_preds[:, 1], test_preds
    
    def fit_pca(self, X, **pca_params):
        scaler = StandardScaler()
        X      = scaler.fit_transform(X)

        pca = PCA(**pca_params)
        pca.fit(X)

        self.scaler = scaler
        self.pca    = pca

        return pca
    
    def fit_transform_tsne(self, X):
        tsne = TSNE(n_jobs=8)
        return tsne.fit_transform(X)

    def transform_pca(self, X):
        X = self.scaler.transform(X)
        return self.pca.transform(X)    
    
    def rf_fi(self, X, y, SEED):
        rf_params = {
            'n_estimators': 500,
            'max_depth': 12,
            'max_features': 'sqrt',
            'min_samples_leaf': 3,
            'random_state': SEED,
            'n_jobs': -1
        }

        model = RandomForestClassifier(**rf_params)

        # preprocess for RF
        for col in X.columns:
            # replace inf with np.nan
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            
            # fill missing values with median
            if X[col].isnull().sum():
                if pd.isnull(X[col].median()):
                    X[col] = X[col].fillna(-1)
                else:
                    X[col] = X[col].fillna(X[col].median())


        model.fit(X, y)
        
        fi_df = pd.DataFrame({
            'feat': X.columns.tolist(),
            'imp': model.feature_importances_
        })

        return fi_df

    
class CategoricalMeanEncoded(BaseEstimator, ClassifierMixin):

    def __init__(self, categorical_features, C=100):
        self.C = C
        self.categorical_features = categorical_features

    def fit(self, df, y=None):
        train_cat    = df.loc[:, self.categorical_features]
        train_target = df['TARGET']
        train_res    = np.zeros((train_cat.shape[0], len(self.categorical_features)), dtype=np.float32)

        self.global_target_mean = train_target.mean()
        self.global_target_std  = train_target.std()

        self.target_sums = {}
        self.target_cnts = {}

        for col in range(len(self.categorical_features)):
            train_res[:, col] = self.fit_transform_column(col, train_target, pd.Series(train_cat.iloc[:, col]))
        
        return train_res

    def predict(self, df):
        test_cat = df.loc[:, self.categorical_features]
        test_res = np.zeros((test_cat.shape[0], len(self.categorical_features)), dtype=np.float32)

        for col in range(len(self.categorical_features)):
            test_res[:, col] = self.transform_column(col, pd.Series(test_cat.iloc[:, col]))
        
        return test_res

    def fit_transform_column(self, col, train_target, train_series):
        self.target_sums[col] = train_target.groupby(train_series).sum()
        self.target_cnts[col] = train_target.groupby(train_series).count()
        
        train_res_neg = self.global_target_mean + self.C
        
        train_res_num = train_series.map(self.target_sums[col]) + train_res_neg
        train_res_den = train_series.map(self.target_cnts[col]) + self.C

        return train_res_num / train_res_den
    
    def transform_column(self, col, test_series):
        test_res_num = test_series.map(self.target_sums[col]).fillna(0.0) + self.global_target_mean * self.C
        test_res_den = test_series.map(self.target_cnts[col]).fillna(0.0) + self.C

        return test_res_num / test_res_den

