import pandas as pd
import numpy as np
import scipy as sp

import argparse
import os
import gc
import time

from base import *
from features import *

from datetime import datetime
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold

basepath = os.path.expanduser('../')

SEED = 1231
np.random.seed(SEED)

#############################################################################################################
#                                       EXPERIMENT PARAMETERS                                               #                                                               
#############################################################################################################

PARAMS = {
    'num_boost_round': 20000,
    'early_stopping_rounds': 200,
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': .03,
    'metric': 'auc',
    'num_leaves': 10,
    'sub_feature': 0.3,
    'min_data_in_leaf': 20,
    'nthread': -1,
    'verbose': -1,
    'seed': SEED
}

MODEL_FILENAME           = 'v135'
SAMPLE_SIZE              = .3

class Modelv135(BaseModel):
    def __init__(self, **params):
        self.params  = params
        self.n_train = 307511 # TODO: find a way to remove this constant
        
    def load_data(self, filenames):
        dfs = []
        
        for filename in filenames:
            dfs.append(np.load(os.path.join(basepath, self.params['output_path'] +  self.params['data_folder'] + f'{filename}')))
        
        dfs  = np.hstack(dfs) # concat across column axis

        df         = pd.DataFrame(dfs)
        
        df.columns = [f'f_{i}' for i in range(dfs.shape[1])] 
        df.index   = np.arange(len(df))

        return df
    
    def reduce_mem_usage(self, df):
        return super(Modelv135, self).reduce_mem_usage(df)
    
    def get_features(self, train, test):
        data       = pd.concat((train, test))
        data.index = np.arange(len(data))
        n_features = data.shape[1]

        t0 = time.time()

        # feature interaction
        # for i in range(n_features):
        #     for j in range(i+1, n_features):
        #         data.loc[:, f'f_{i}{j}'] = data.iloc[:, i] - data.iloc[:, j]
        
        # print('Took: {} seconds to generate feature interactions'.format(time.time() - t0))

        return data

    # This method would perform feature engineering on merged datasets.
    def fe(self, train, test):
        original_train = train.copy()
        data           = self.get_features(original_train, test)

        train = data.iloc[:len(train)]
        test  = data.iloc[len(train):]

        del data, original_train
        gc.collect()

        return train, test

    def train(self, train, test, feature_list, is_eval, TARGET_NAME='TARGET', **params):
        X = train.loc[:, feature_list]
        y = train.loc[:, TARGET_NAME]
        
        Xte = test.loc[:, feature_list]
        yte = []

        if is_eval:
            yte = test.loc[:, TARGET_NAME]
        
        return super(Modelv135, self).train_lgb(X, y, Xte, yte, **params)

    def evaluate(self, test, feature_list, is_eval, model, TARGET_NAME='TARGET'):
        Xte = test.loc[:, feature_list]
        yte = []

        if is_eval:
            yte = test.loc[:, TARGET_NAME]

        return super(Modelv135, self).evaluate_lgb(Xte, yte, model)


    def predict_test(self, train, test, feature_list, params, save_path, n_folds=5):
        return super(Modelv135, self).predict_test_xgb(train, test, feature_list, params, save_path, n_folds=n_folds)


    def cross_validate(self, train, feature_list, params, cv_adversarial_filepath=None, TARGET_NAME='TARGET'):
        Xtr = train.loc[:, feature_list]
        ytr = train.loc[:, TARGET_NAME]

        return super(Modelv135, self).cross_validate(Xtr, ytr, params, cv_adversarial_filepath=cv_adversarial_filepath)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Home Credit Default Risk Solution')
    
    parser.add_argument('-input_path', help='Path to input directory')     # path to raw files
    parser.add_argument('-output_path', help='Path to output directory')   # path to working data folder 
    parser.add_argument('-data_folder', help='Folder name of the dataset') # dataset folder name
    parser.add_argument('-cv', type=bool, help='Cross Validation')
    parser.add_argument('-t',type=bool, help='Full Training on a given seed.')
    parser.add_argument('-s', type=bool, help='Whether to work on a sample or not.')
    parser.add_argument('-seed', type=int, help='Random SEED')
    parser.add_argument('-cv_seed', type=int, help='CV SEED')
    
    args    = parser.parse_args()

    if args.cv:
        print('Cross validation on training and store parameters and cv score on disk ...')
        
        train_filenames = [
                            'v127_4457_oof_train_preds.npy',
                            'v128_4457_oof_train_preds.npy',
                            'v129_4457_oof_train_preds.npy',
                            'v131_4457_oof_train_preds.npy',
                            'v132_4457_oof_train_preds.npy',
                            'v133_4457_oof_train_preds.npy',
                            'v134_4457_oof_train_preds.npy'
                          ]

        test_filenames  = [
                            'v127_4457_test_preds.npy',
                            'v128_4457_test_preds.npy',
                            'v129_4457_test_preds.npy',
                            'v131_4457_test_preds.npy',
                            'v132_4457_test_preds.npy',
                            'v133_4457_test_preds.npy',
                            'v134_4457_test_preds.npy'
                          ]

        input_path      = args.input_path
        output_path     = args.output_path
        data_folder     = args.data_folder
        is_sample       = args.s
        SEED            = args.seed

        params = {
            'input_path': input_path,
            'output_path': output_path,
            'data_folder': data_folder
        }

        m   = Modelv135(**params)
            
        train  = m.load_data(train_filenames)
        test   = m.load_data(test_filenames)

        train, test  = m.fe(train, test)

        # load target
        target = pd.read_pickle(os.path.join(basepath, output_path + 'feature_groups/' + f'application_train.pkl'))['TARGET']       
        train.loc[:, 'TARGET'] = target.values # add target to train

        data   = pd.concat((train, test))
        data   = m.reduce_mem_usage(data)

        print('Shape of data: {}'.format(data.shape))
    
        train  = data.iloc[:m.n_train]

        del data, test
        gc.collect()

        feature_list = train.columns.drop('TARGET').tolist()
        
        PARAMS['seed']                  = SEED
        PARAMS['feature_fraction_seed'] = SEED
        PARAMS['bagging_seed']          = SEED
        
        cv_adversarial_filepath = None

        cv_history = m.cross_validate(train, feature_list, PARAMS.copy(), cv_adversarial_filepath)
        cv_score   = str(cv_history.iloc[-1]['auc-mean']) + '_' + str(cv_history.iloc[-1]['auc-stdv'])
        
        PARAMS['num_boost_round'] = len(cv_history)

        print('*' * 100)
        print('Best AUC: {}'.format(cv_score))
        
        joblib.dump(PARAMS, os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{SEED}_params.pkl'))
        joblib.dump(cv_score, os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{SEED}_cv.pkl'))
    
    elif args.t:
        print('Full training ..')

        train_filenames = [
                            'v127_4457_oof_train_preds.npy',
                            'v128_4457_oof_train_preds.npy',
                            'v129_4457_oof_train_preds.npy',
                            'v131_4457_oof_train_preds.npy',
                            'v132_4457_oof_train_preds.npy',
                            'v133_4457_oof_train_preds.npy',
                            'v134_4457_oof_train_preds.npy'
                          ]

        test_filenames  = [
                            'v127_4457_test_preds.npy',
                            'v128_4457_test_preds.npy',
                            'v129_4457_test_preds.npy',
                            'v131_4457_test_preds.npy',
                            'v132_4457_test_preds.npy',
                            'v133_4457_test_preds.npy',
                            'v134_4457_test_preds.npy'
                          ]


        input_path      = args.input_path
        output_path     = args.output_path
        data_folder     = args.data_folder
        is_sample       = args.s
        CV_SEED         = args.cv_seed
        SEED            = args.seed

        params = {
            'input_path': input_path,
            'output_path': output_path,
            'data_folder': data_folder
        }

        m   = Modelv135(**params)
            
        train  = m.load_data(train_filenames)
        test   = m.load_data(test_filenames)

        train, test  = m.fe(train, test)

        # load target
        target = pd.read_pickle(os.path.join(basepath, output_path + 'feature_groups/' + f'application_train.pkl'))['TARGET']       
        train.loc[:, 'TARGET'] = target.values # add target to train

        data   = pd.concat((train, test))
        data   = m.reduce_mem_usage(data)

        print('Shape of data: {}'.format(data.shape))
    
        train  = data.iloc[:m.n_train]
        test   = data.iloc[m.n_train:]

        del data
        gc.collect()

        feature_list = train.columns.drop('TARGET').tolist()

        # Load params and holdout score from disk.
        PARAMS        = joblib.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_params.pkl'))
        HOLDOUT_SCORE = joblib.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_cv.pkl'))

        PARAMS['seed'] = SEED

        print('*' * 100)
        print('PARAMS are: {}'.format(PARAMS))

        # train model
        model, feat_df = m.train(train, test, feature_list, is_eval=False, **PARAMS)
        
        # evaluation part
        preds, score  = m.evaluate(test, feature_list, is_eval=False, model=model)

        sub_identifier = "%s-%s-%s-%s-%s" % (datetime.now().strftime('%Y%m%d-%H%M'), MODEL_FILENAME, HOLDOUT_SCORE, SEED, data_folder[:-1])

        sub            = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv.zip'))
        sub['TARGET']  = preds

        sub.to_csv(os.path.join(basepath, 'submissions/%s.csv'%(sub_identifier)), index=False)