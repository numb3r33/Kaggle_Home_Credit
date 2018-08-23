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
    'num_leaves': 8,
    'sub_feature': 0.3,
    'min_data_in_leaf': 20,
    'nthread': 8,
    'verbose': -1,
    'seed': SEED
}

MODEL_FILENAME           = 'v126'
SAMPLE_SIZE              = .3

class Modelv126(BaseModel):
    def __init__(self, **params):
        self.params  = params
        self.n_train = 307511 # TODO: find a way to remove this constant
        
    def load_data(self, filenames):
        dfs = []
        
        for filename in filenames:
            dfs.append(np.load(os.path.join(basepath, self.params['output_path'] +  self.params['data_folder'] + f'{filename}')))
        
        dfs  = np.hstack(dfs) # concat across column axis

        df       = pd.DataFrame(dfs, columns=[f'f_{i}' for i in range(dfs.shape[1])])
        df.index = np.arange(len(df))

        return df
    
    def reduce_mem_usage(self, df):
        return super(Modelv126, self).reduce_mem_usage(df)
    
    def get_features(self, train, test):
        data       = pd.concat((train, test))
        data.index = np.arange(len(data))

        # feature interaction
        for i in range(data.shape[1]):
            for j in range(i+1, data.shape[1]):
                data.loc[:, f'f_{i}{j}'] = data[i] - data[j]
        
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


    def predict_test(self, train, test, feature_list, params, save_path, n_folds=5):
        return super(Modelv126, self).predict_test_xgb(train, test, feature_list, params, save_path, n_folds=n_folds)


    def cross_validate(self, train, feature_list, params, cv_adversarial_filepath=None, TARGET_NAME='TARGET'):
        Xtr = train.loc[:, feature_list]
        ytr = train.loc[:, TARGET_NAME]

        return super(Modelv126, self).cross_validate_lgb(Xtr, ytr, params, cv_adversarial_filepath=cv_adversarial_filepath)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Home Credit Default Risk Solution')
    
    parser.add_argument('-input_path', help='Path to input directory')     # path to raw files
    parser.add_argument('-output_path', help='Path to output directory')   # path to working data folder 
    parser.add_argument('-data_folder', help='Folder name of the dataset') # dataset folder name
    parser.add_argument('-cv', type=bool, help='Cross Validation')
    parser.add_argument('-cv_predict',type=bool, help='Cross Validation and Predictions for test set.')
    parser.add_argument('-s', type=bool, help='Whether to work on a sample or not.')
    parser.add_argument('-seed', type=int, help='Random SEED')
    parser.add_argument('-cv_seed', type=int, help='CV SEED')
    
    args    = parser.parse_args()

    if args.cv:
        print('Cross validation on training and store parameters and cv score on disk ...')
        
        train_filenames = ['v123_4457_oof_train_preds.npy',
                           'v124_4457_oof_train_preds.npy'
                          ]

        test_filenames  = []

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

        m   = Modelv126(**params)
            
        train  = m.load_data(train_filenames)
        test   = m.load_data(test_filenames)
        
        train, test  = m.fe(train, test)
        
        data   = pd.concat((train, test))
        data   = m.reduce_mem_usage(data)

        print('Shape of data: {}'.format(data.shape))
    
        train  = data.iloc[:m.n_train]

        del data, test
        gc.collect()

        feature_list = data.columns.drop('TARGET').tolist()
        
        PARAMS['seed']                  = SEED
        PARAMS['feature_fraction_seed'] = SEED
        PARAMS['bagging_seed']          = SEED
        
        cv_adversarial_filepath = os.path.join(basepath, 'data/raw/cv_idx_test_stratified.csv')

        cv_history = m.cross_validate(train, feature_list, PARAMS.copy(), cv_adversarial_filepath)
        cv_score   = str(cv_history.iloc[-1]['auc-mean']) + '_' + str(cv_history.iloc[-1]['auc-stdv'])
        
        PARAMS['num_boost_round'] = len(cv_history)

        print('*' * 100)
        print('Best AUC: {}'.format(cv_score))
        
        joblib.dump(PARAMS, os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{SEED}_params.pkl'))
        joblib.dump(cv_score, os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{SEED}_cv.pkl'))
    
    elif args.cv_predict:
        print('Cross validation with different seeds and produce submission for test set ..')

        input_path      = args.input_path
        output_path     = args.output_path
        data_folder     = args.data_folder
        is_sample       = args.s
        SEED            = args.seed
        CV_SEED         = args.cv_seed

        params = {
            'input_path': input_path,
            'output_path': output_path,
            'data_folder': data_folder
        }

        m   = Modelv126(**params)
        

        # Loading data
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}data.h5')):
            print('Loading dataset from disk ...')
            data = pd.read_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')
        else:
            print('Merge feature groups and save them to disk ...')
            train, test  = m.merge_datasets()
            train, test  = m.fe(train, test, compute_categorical='ohe')
            
            data         = pd.concat((train, test))
            data         = m.reduce_mem_usage(data)

            data.to_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')

            del train, test
            gc.collect()

        train  = data.iloc[:m.n_train]
        test   = data.iloc[m.n_train:]

        del data
        gc.collect()
        
        # Generating a sample if required
        if is_sample:
            print('*' * 100)
            print('Take a random sample of the training data ...')
            train = train.sample(frac=SAMPLE_SIZE)
        
        # check to see if feature list exists on disk or not for a particular model
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy')):
            feature_list = np.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy'))
        else: 
            feature_list = train.columns.tolist()
            feature_list = list(set(feature_list) - set(COLS_TO_REMOVE))
            np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy'), feature_list)

        
        PARAMS['seed']                  = SEED
        PARAMS['feature_fraction_seed'] = SEED
        PARAMS['bagging_seed']          = SEED

        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_test_preds.npy')):
            oof_train_preds = np.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_oof_train_preds.npy'))       
            test_preds      = np.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_test_preds.npy'))
            auc             = joblib.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_oof_auc.pkl'))
        else:
            save_path = os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}')
            auc, oof_train_preds, test_preds = m.predict_test(train, test, feature_list, PARAMS.copy(), save_path)

            np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_oof_train_preds.npy'), oof_train_preds)
            np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_test_preds.npy'), test_preds)
            joblib.dump(auc, os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_oof_auc.pkl'))
        
        sub_identifier = "%s-%s-%s-%s-%s" % (datetime.now().strftime('%Y%m%d-%H%M'), MODEL_FILENAME, auc, SEED, data_folder[:-1])

        # generate for test set
        sub            = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv.zip'))
        sub['TARGET']  = test_preds
        sub.to_csv(os.path.join(basepath, 'submissions/%s.csv'%(sub_identifier)), index=False)