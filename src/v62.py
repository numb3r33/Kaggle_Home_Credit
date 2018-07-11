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

COLS_TO_REMOVE = ['SK_ID_CURR', 
                  'TARGET',
                  'OCCUPATION_TYPE__5',
                  'OCCUPATION_TYPE__-1',
                  'OCCUPATION_TYPE__11',
                  'OCCUPATION_TYPE__15',
                  'ORGANIZATION_TYPE__29',
                  'ORGANIZATION_TYPE__5',
                  'FLAG_OWN_REALTY',
                  'FLAG_DOCUMENT_21',
                  'ORGANIZATION_TYPE__21',
                  'FLAG_DOCUMENT_14',
                  'ORGANIZATION_TYPE__17',
                  'ORGANIZATION_TYPE__27',
                  'ORGANIZATION_TYPE__32',
                  'FLAG_DOCUMENT_16',
                  'ORGANIZATION_TYPE__47',
                  'FLAG_DOCUMENT_13',
                  'FLAG_DOCUMENT_11',
                  'ORGANIZATION_TYPE__40',
                  'ORGANIZATION_TYPE__23',
                  'ORGANIZATION_TYPE__14',
                  'diff_max_min_credit_term',
                  'ORGANIZATION_TYPE__1',
                  'ORGANIZATION_TYPE__9',
                  'OCCUPATION_TYPE__nan',
                  'ORGANIZATION_TYPE__41',
                  'OCCUPATION_TYPE__7',
                  'FLAG_MOBIL',
                  'ORGANIZATION_TYPE__18',
                  'ORGANIZATION_TYPE__38',
                  'ORGANIZATION_TYPE__44',
                  'FLAG_DOCUMENT_12',
                  'ORGANIZATION_TYPE__0',
                  'FLAG_DOCUMENT_2',
                  'ORGANIZATION_TYPE__13',
                  'OCCUPATION_TYPE__0',
                  'FLAG_DOCUMENT_4',
                  'OCCUPATION_TYPE__16',
                  'ORGANIZATION_TYPE__49',
                  'FLAG_DOCUMENT_6',
                  'FLAG_DOCUMENT_9',
                  'ORGANIZATION_TYPE__nan',
                  'OCCUPATION_TYPE__12',
                  'ORGANIZATION_TYPE__20',
                  'FLAG_CONT_MOBILE',
                  'ORGANIZATION_TYPE__37',
                  'ORGANIZATION_TYPE__45',
                  'FLAG_EMP_PHONE',
                  'FLAG_DOCUMENT_17',
                  'LIVE_REGION_NOT_WORK_REGION',
                  'OCCUPATION_TYPE__17',
                  'NAME_TYPE_SUITE',
                  'ORGANIZATION_TYPE__15',
                  'REG_REGION_NOT_LIVE_REGION',
                  'FLAG_DOCUMENT_10',
                  'ORGANIZATION_TYPE__3',
                  'OCCUPATION_TYPE__2',
                  'ORGANIZATION_TYPE__19',
                  'FLAG_DOCUMENT_19',
                  'AMT_REQ_CREDIT_BUREAU_DAY',
                  'credits_ended_bureau',
                  'ORGANIZATION_TYPE__8',
                  'ORGANIZATION_TYPE__16',
                  'FLAG_DOCUMENT_8',
                  'ORGANIZATION_TYPE__25',
                  'OCCUPATION_TYPE__6',
                  'NUM_NULLS_EXT_SCORES',
                  'ORGANIZATION_TYPE__48',
                  'ORGANIZATION_TYPE__53',
                  'ORGANIZATION_TYPE__10',
                  'FLAG_DOCUMENT_7',
                  'ORGANIZATION_TYPE__55',
                  'ORGANIZATION_TYPE__24',
                  'NAME_EDUCATION_TYPE__0',
                  'ORGANIZATION_TYPE__46',
                  'ELEVATORS_MODE',
                  'NAME_EDUCATION_TYPE__nan',
                  'ORGANIZATION_TYPE__22',
                  'ORGANIZATION_TYPE__50',
                  'REG_REGION_NOT_WORK_REGION',
                  'ORGANIZATION_TYPE__56',
                  'FLAG_DOCUMENT_5',
                  'FLAG_DOCUMENT_20',
                  'ORGANIZATION_TYPE__2',
                  'ORGANIZATION_TYPE__6',
                  'OCCUPATION_TYPE__13',
                  'ORGANIZATION_TYPE__52',
                  'FLAG_DOCUMENT_15',
                  'ORGANIZATION_TYPE__43',
                  'AMT_REQ_CREDIT_BUREAU_HOUR',
                  'NAME_HOUSING_TYPE',
                  'ORGANIZATION_TYPE__11',
                  'HOUSETYPE_MODE',
                  'EMERGENCYSTATE_MODE',
                  'ORGANIZATION_TYPE__28',
                  'NAME_EDUCATION_TYPE__2',
                  'ORGANIZATION_TYPE__4',
                  'OCCUPATION_TYPE__14',
                  'ORGANIZATION_TYPE__35',
                  'LIVE_CITY_NOT_WORK_CITY',
                  'num_diff_credits',
                  'ORGANIZATION_TYPE__51',
                  'REG_CITY_NOT_WORK_CITY',
                  'FLAG_EMAIL',
                  'ORGANIZATION_TYPE__57',
                  'NAME_HOUSING_TYPE__0',
                  'NAME_INCOME_TYPE__2',
                  'NAME_INCOME_TYPE__5',
                  'NAME_HOUSING_TYPE__nan',
                  'NAME_INCOME_TYPE__nan',
                  'NAME_INCOME_TYPE__0',
                  'NAME_INCOME_TYPE__6',
                  'NAME_CONTRACT_STATUS_3',
                  'NAME_INCOME_TYPE__3',
                  'diff_balance_curr_credit',
                  'ratio_min_installment_balance',
                  'NAME_HOUSING_TYPE__4',
                  'CODE_REJECT_REASON_5',
                  'CODE_REJECT_REASON_8',
                  'ORGANIZATION_TYPE__33',
                  'CODE_REJECT_REASON_0',
                  'OCCUPATION_TYPE__1',
                  'NAME_HOUSING_TYPE__5',
                  'sum_num_times_prolonged',
                  'NAME_GOODS_CATEGORY_13',
                  'NAME_GOODS_CATEGORY_4',
                  'NAME_GOODS_CATEGORY_26',
                  'PRODUCT_COMBINATION_-1',
                  'NAME_GOODS_CATEGORY_24',
                  'NAME_GOODS_CATEGORY_15',
                  'NAME_GOODS_CATEGORY_20',
                  'NAME_GOODS_CATEGORY_9',
                  'CODE_REJECT_REASON_6',
                  'NAME_GOODS_CATEGORY_6',
                  'NAME_GOODS_CATEGORY_0',
                  'num_high_int_no_info_loans',
                  'NAME_HOUSING_TYPE__2',
                  'NAME_GOODS_CATEGORY_14',
                  'NAME_GOODS_CATEGORY_17',
                  'PRODUCT_COMBINATION_16',
                  'PRODUCT_COMBINATION_15',
                  'OCCUPATION_TYPE__10',
                  'PRODUCT_COMBINATION_14',
                  'NAME_GOODS_CATEGORY_1',
                  'NAME_GOODS_CATEGORY_12',
                  'NAME_GOODS_CATEGORY_21',
                  'NAME_GOODS_CATEGORY_25',
                  'OCCUPATION_TYPE__9',
                  'NAME_GOODS_CATEGORY_10',
                  'NAME_GOODS_CATEGORY_16',
                  'NAME_GOODS_CATEGORY_8'
                ] #+ [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]

LGB_PARAMS = {
    'num_boost_round': 10000,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': .03,
    'metric': 'auc',
    'max_depth': 7,
    'num_leaves': 60,
    'sub_feature': .1,
    'sub_row': .7,
    'feature_fraction_seed': SEED,
    'bagging_seed': SEED,
    'min_data_in_leaf': 60,
    'nthread': 4,
    'verbose': -1,
    'seed': SEED
}


XGB_PARAMS = {
    'num_boost_round': 10000,    
    'booster': 'gbtree',
    'silent': 1,
    'nthread': 4,
    'eta': .05,
    'max_depth': 2,
    'colsample_bytree': .2,
    'colsample_bylevel': .3,
    'objective': 'binary:logistic',
    'min_child_weight': 1,
    'eval_metric': 'auc',
    'seed': SEED
}

MODEL_FILENAME = 'v62_model.txt'
SAMPLE_SIZE    = .5

# NOTE: column in frequency encoded columns
# cannot be in ohe cols.
FREQ_ENCODING_COLS = ['ORGANIZATION_OCCUPATION',
                      'age_emp_categorical',
                      'age_occupation'
                     ]

OHE_COLS           = [
                      'ORGANIZATION_TYPE',
                      'OCCUPATION_TYPE',
                      'NAME_EDUCATION_TYPE',
                      'NAME_HOUSING_TYPE',
                      'NAME_INCOME_TYPE'
                     ]


class Modelv62(BaseModel):
    def __init__(self, **params):
        self.params  = params
        self.n_train = 307511 # TODO: find a way to remove this constant
    
    def load_data(self, filenames):
        dfs = []
        
        for filename in filenames:
            dfs.append(pd.read_csv(filename, parse_dates=True, keep_date_col=True))
        
        df       = pd.concat(dfs)
        df.index = np.arange(len(df))
        df       = super(Modelv62, self).reduce_mem_usage(df)

        return df	

    def preprocess(self):
        
        # application train and application test
        app_train_filename = os.path.join(basepath, self.params['input_path'] + 'application_train.csv.zip')
        app_test_filename  = os.path.join(basepath, self.params['input_path'] + 'application_test.csv.zip')
        
        df   = self.load_data([app_train_filename, app_test_filename])

        # save to disk
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'])):
            os.mkdir(os.path.join(basepath, self.params['output_path'] + self.params['run_name']))

        df.iloc[:self.n_train].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'application_train.pkl'))
        df.iloc[self.n_train:].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'application_test.pkl'))
        
        del df
        gc.collect()

        # bureau
        bureau_filename = os.path.join(basepath, self.params['input_path'] + 'bureau.csv.zip')
        df  = self.load_data([bureau_filename])
        df.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'bureau.pkl'))

        del df
        gc.collect()

        # bureau balance
        bureau_balance_filename = os.path.join(basepath, self.params['input_path'] + 'bureau_balance.csv.zip')
        df = self.load_data([bureau_balance_filename])
        df.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'bureau_balance.pkl'))

        del df
        gc.collect()

        # credit card balance
        credit_card_filename = os.path.join(basepath, self.params['input_path'] + 'credit_card_balance.csv.zip')
        df = self.load_data([credit_card_filename])
        df.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'credit_card_balance.pkl'))

        del df
        gc.collect()

        # installments payments
        installments_payments_filename = os.path.join(basepath, self.params['input_path'] + 'installments_payments.csv.zip')
        df = self.load_data([installments_payments_filename])
        df.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'installments_payments.pkl'))

        del df
        gc.collect()

        # POS_CASH_balance
        pos_cash_balance_filename = os.path.join(basepath, self.params['input_path'] + 'POS_CASH_balance.csv.zip')
        df = self.load_data([pos_cash_balance_filename])
        df.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'POS_CASH_balance.pkl'))

        del df
        gc.collect()

        # previous_application
        previous_application_filename = os.path.join(basepath, self.params['input_path'] + 'previous_application.csv.zip')
        df = self.load_data([previous_application_filename])
        df.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'previous_application.pkl'))

        del df
        gc.collect()

    def create_folds(self, fold_name, seed):
        data     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'application_train.pkl'))
        dtr, dte = super(Modelv62, self).create_fold(data, seed)

        dtr.index = np.arange(len(dtr))
        dte.index = np.arange(len(dte))

        dtr.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'application_{fold_name}train.pkl'))
        dte.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'application_{fold_name}test.pkl'))

    # TODO: not super happy with the way features are constructed right now
    # because for every type of feature we are first generating those features and then persisting 
    # them to disk.
    # Possible Solution: Provide data source and feature and let a layer decide based on feature
    # type how to create and store them on the disk, that layer should be responsible to load up features given
    # feature type.

    def prepare_features(self, fold_name):
        tr = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'application_{fold_name}train.pkl'))
        te = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'application_{fold_name}test.pkl'))
        ntrain = len(tr)

        data = pd.concat((tr, te))
        
        del tr, te
        gc.collect()

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'current_application_{fold_name}train.pkl')):
            print('Generating features based on current application ....')

            t0 = time.clock()
            data, FEATURE_NAMES = current_application_features(data)
            data.index = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'current_application_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'current_application_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))
        else:
            print('Already generated features based on current application')


        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'bureau_{fold_name}train.pkl')):
            bureau = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'bureau.pkl'))
            
            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes

            print('Generating features based on credits reported to bureau ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = bureau_features(bureau, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'bureau_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'bureau_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

            del bureau
            gc.collect()

        else:
            print('Already generated features based on bureau application')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'bureau_bal_{fold_name}train.pkl')):
            bureau     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'bureau.pkl'))
            bureau_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'bureau_balance.pkl')) 

            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes
            
            for col in bureau_bal.select_dtypes(include=['category']).columns:
                bureau_bal.loc[:, col] = bureau_bal.loc[:, col].cat.codes

            print('Generating features based on credits reported to bureau and bureau balance ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = bureau_and_balance(bureau, bureau_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'bureau_bal_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'bureau_bal_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on bureau and balance')


        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_{fold_name}train.pkl')):
            prev_app = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'previous_application.pkl')) 

            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            print('Generating features based on previous application ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_features(prev_app, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            del prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on previous application')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'pos_cash_{fold_name}train.pkl')):
            pos_cash = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'POS_CASH_balance.pkl')) 

            for col in pos_cash.select_dtypes(include=['category']).columns:
                pos_cash.loc[:, col] = pos_cash.loc[:, col].cat.codes
            
            print('Generating features based on pos cash ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = pos_cash_features(pos_cash, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            del pos_cash
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'pos_cash_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'pos_cash_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on pos cash')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'credit_{fold_name}train.pkl')):
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'credit_card_balance.pkl')) 

            for col in credit_bal.select_dtypes(include=['category']).columns:
                credit_bal.loc[:, col] = credit_bal.loc[:, col].cat.codes
            
            print('Generating features based on Credit Card ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = credit_card_features(credit_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            del credit_bal
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'credit_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'credit_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Credit Card')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'installments_{fold_name}train.pkl')):
            installments = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'installments_payments.pkl')) 

            for col in installments.select_dtypes(include=['category']).columns:
                installments.loc[:, col] = installments.loc[:, col].cat.codes
            
            print('Generating features based on Installments ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = get_installment_features(installments, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            del installments
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'installments_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'installments_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Installments')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_bureau_{fold_name}train.pkl')):
            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'previous_application.pkl')) 
            bureau     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'bureau.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes
            
            
            print('Generating features based on Previous Applications and Bureau Applications....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_bureau(prev_app, bureau, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            del bureau, prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_bureau_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_bureau_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Previous application and Bureau Applications')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_credit_{fold_name}train.pkl')):
            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'previous_application.pkl')) 
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'credit_card_balance.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            for col in credit_bal.select_dtypes(include=['category']).columns:
                credit_bal.loc[:, col] = credit_bal.loc[:, col].cat.codes
            
            
            print('Generating features based on Previous Applications and Credit card balance ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_credit_card(prev_app, credit_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            del credit_bal, prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_credit_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_credit_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Previous application and Credit card balance')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_installments_{fold_name}train.pkl')):
            prev_app     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'previous_application.pkl')) 
            installments = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'installments_payments.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            for col in installments.select_dtypes(include=['category']).columns:
                installments.loc[:, col] = installments.loc[:, col].cat.codes
            
            
            print('Generating features based on Previous Applications and Installment Payments ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_installments(prev_app, installments, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            del installments, prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_installments_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_installments_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Previous application and Installment Payments.')
        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'loan_stacking_{fold_name}train.pkl')):
            bureau     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'bureau.pkl'))
            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'previous_application.pkl')) 
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'credit_card_balance.pkl')) 
            
            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes

            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            for col in credit_bal.select_dtypes(include=['category']).columns:
                credit_bal.loc[:, col] = credit_bal.loc[:, col].cat.codes


            print('Generating features based on loan stacking ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = loan_stacking(bureau, prev_app, credit_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'loan_stacking_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'loan_stacking_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

            del bureau
            gc.collect()
        else:
            print('Already generated features based on loan stacking.')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'feature_groups_{fold_name}train.pkl')):
            print('Generating features based on feature groups ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = feature_groups(data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'feature_groups_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'feature_groups_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on feature groups.')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_pos_cash_{fold_name}train.pkl')):
            print('Generating features based on previous application and pos cash ....')

            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'previous_application.pkl')) 
            pos_cash   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'POS_CASH_balance.pkl')) 

            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes

            for col in pos_cash.select_dtypes(include=['category']).columns:
                pos_cash.loc[:, col] = pos_cash.loc[:, col].cat.codes
            
            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_pos(prev_app, pos_cash, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_pos_cash_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_pos_cash_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on previous application and pos cash.')


        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_pos_cash_credit_bal_{fold_name}train.pkl')):
            print('Generating features based on previous application, pos cash and credit card balance ....')

            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'previous_application.pkl')) 
            pos_cash   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'POS_CASH_balance.pkl')) 
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'credit_card_balance.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes

            for col in pos_cash.select_dtypes(include=['category']).columns:
                pos_cash.loc[:, col] = pos_cash.loc[:, col].cat.codes
            
            for col in credit_bal.select_dtypes(include=['category']).columns:
                credit_bal.loc[:, col] = credit_bal.loc[:, col].cat.codes

            
            t0                  = time.time()
            data, FEATURE_NAMES = prev_app_pos_credit(prev_app, pos_cash, credit_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_pos_cash_credit_bal_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_pos_cash_credit_bal_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.time() - t0))

        else:
            print('Already generated features based on previous application, pos cash and credit card balance.')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_ohe_{fold_name}train.pkl')):
            print('Generating features based on previous application one hot encoded features ....')

            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'previous_application.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            t0                  = time.time()
            data, FEATURE_NAMES = prev_app_ohe(prev_app, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv62, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_ohe_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_ohe_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.time() - t0))

        else:
            print('Already generated features based on previous application one hot encode features.')

        
    # This method currently takes care of loading engineered features from disk
    # and merging train and test to report back a dataframe (data) which can be used by
    # other layers.
    def merge_datasets(self, fold_name, test_df=False):

        def get_filenames(fold_name):
            filenames = [f'application_{fold_name}',
                        f'current_application_{fold_name}',
                        f'bureau_{fold_name}',
                        f'prev_app_{fold_name}',
                        f'pos_cash_{fold_name}',
                        f'credit_{fold_name}',					 
                        f'installments_{fold_name}',					 
                        f'prev_app_bureau_{fold_name}',					 
                        f'prev_app_credit_{fold_name}',				 
                        f'prev_app_installments_{fold_name}',
                        f'loan_stacking_{fold_name}',
                        f'feature_groups_{fold_name}',
                        f'prev_app_pos_cash_{fold_name}',
                        f'prev_app_pos_cash_credit_bal_{fold_name}',
                        f'prev_app_ohe_{fold_name}'                                                      
                        ]

            return filenames

        train     = []
        test      = []
        full_test = []

        filenames = get_filenames(fold_name)
        for filename_ in filenames:
            tmp       = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'{filename_}train.pkl'))
            tmp.index = np.arange(len(tmp)) 
            train.append(tmp)

        for filename_ in filenames:
            tmp       = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'{filename_}test.pkl'))
            tmp.index = np.arange(len(tmp))
            test.append(tmp)
        
        if test_df:
            fold_name = '' # to represent test set
            filenames = get_filenames(fold_name)
            
            for filename_ in filenames:
                tmp       = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'{filename_}test.pkl'))
                tmp.index = np.arange(len(tmp))
                full_test.append(tmp)
        

        return pd.concat(train, axis=1), pd.concat(test, axis=1), pd.concat(full_test, axis=1) if len(full_test) else None

    # This method just calls the base class with X,y, Xte and yte in the right format
    # to train and returns a trained model which could be dumped on disk for further use.
    # TODO: Find out why we are not able to load back model from disk and generate correct predictions
    # there seems to be some issue in it right now.
    def train(self, train, test, feature_list, is_eval, TARGET_NAME='TARGET', **params):
        X = train.loc[:, feature_list]
        y = train.loc[:, TARGET_NAME]
        
        Xte = test.loc[:, feature_list]
        yte = []

        if is_eval:
            yte = test.loc[:, TARGET_NAME]
        
        return super(Modelv62, self).train_lgb(X, y, Xte, yte, **params) if self.params['model_type'] == 'LGB'\
               else super(Modelv62, self).train_xgb(X, y, Xte, yte, **params)

    # This method just takes in a model and test dataset and returns predictions 
    # prints out AUC on the test dataset as well in the process.
    def evaluate(self, test, feature_list, is_eval, model, TARGET_NAME='TARGET'):
        Xte = test.loc[:, feature_list]
        yte = []

        if is_eval:
            yte = test.loc[:, TARGET_NAME]

        return super(Modelv62, self).evaluate_lgb(Xte, yte, model) if self.params['model_type'] == 'LGB'\
               else super(Modelv62, self).evaluate_xgb(Xte, yte, model)

    # This method just takes a stratified sample of the training dataset and returns
    # back the sample.
    # TODO: add a parameter to switch between stratified and random sampling.
    def get_sample(self, train, sample_size=.5, seed=SEED):
        """
        Generates a stratified sample of provided sample_size
        """
        return super(Modelv62, self).get_sample(train, sample_size, seed=SEED)

    # z-standardization
    def feature_transformation(self, data):
        # print('sample feature\n')
        # print(data['EXT_1_2_sum'].describe())
        # print()
        
        for col in data.select_dtypes(include=['float16', 'float32']).columns:
            if col == 'TARGET':
                continue
            mean_value = data[col].mean()

            if pd.notnull(mean_value):
                data.loc[:, col] = data[col] - mean_value
                
        return data 

    def get_features(self, train, test, compute_ohe):
        data       = pd.concat((train, test))
        data.index = np.arange(len(data))

        for col in data.select_dtypes(include=['category']).columns:
            data[col] = data[col].cat.codes

        # TODO: not very happy with the way we are computing interactions
        # because if we omit any of this feature from pipeline it would
        # still work but would most likely be a feature of all null values.

        # concatenate OCCUPATION TYPE AND ORGANIZATION TYPE
        data.loc[:, 'ORGANIZATION_OCCUPATION'] = pd.factorize(data.ORGANIZATION_TYPE.astype(np.str) +\
                                                              data.OCCUPATION_TYPE.astype(np.str)
                                                             )[0]
        
        # interaction between total debt to income and (annuity / credit)
        data.loc[:, 'debt_income_to_annuity_credit'] = data.total_debt_to_income / data.ratio_annuity_credit

        # interaction between days birth and ratio of annuity to credit
        data.loc[:, 'add_days_birth_annuity_credit'] = data.DAYS_BIRTH + data.ratio_annuity_credit

        # interaction between ratio of annuity to credit with external source 2 score
        data.loc[:, 'mult_annuity_credit_ext_source_2']  = data.ratio_annuity_credit * data.EXT_SOURCE_2
        data.loc[:, 'ratio_annuity_credit_ext_source_2'] = data.ratio_annuity_credit / data.EXT_SOURCE_2.map(np.log1p)

        data.loc[:, 'mult_annuity_credit_ext_source_1']  = data.ratio_annuity_credit * data.EXT_SOURCE_1
        data.loc[:, 'ratio_annuity_credit_ext_source_1'] = data.ratio_annuity_credit / data.EXT_SOURCE_1.map(np.log1p)
        
        data.loc[:, 'mult_annuity_credit_ext_source_3']  = data.ratio_annuity_credit * data.EXT_SOURCE_3
        data.loc[:, 'ratio_annuity_credit_ext_source_3'] = data.ratio_annuity_credit / data.EXT_SOURCE_3.map(np.log1p)
        

        # interaction between ratio of annuity to credit with total amount paid in installments
        data.loc[:, 'mult_annuity_credit_amt_payment_sum'] = data.ratio_annuity_credit * data.AMT_PAYMENT_sum

        # interaction between total amount paid in installments and delay in installments
        data.loc[:, 'mult_amt_payment_sum_delay_installment'] = data.AMT_PAYMENT_sum * data.delay_in_installment_payments

        # interaction between credit / annuity and age
        data.loc[:, 'diff_credit_annuity_age'] = (data.AMT_CREDIT / data.AMT_ANNUITY) - (-data.DAYS_BIRTH / 365)

        # interaction between ext_3 and age
        data.loc[:, 'ext_3_age'] = data.EXT_SOURCE_3 * (-data.DAYS_BIRTH / 365)

        # interaction between ext_2 and age
        data.loc[:, 'ext_2_age'] = data.EXT_SOURCE_2 * (-data.DAYS_BIRTH / 365)

        # interaction between rate and external source 2
        data.loc[:, 'add_rate_ext_2'] = (data.AMT_CREDIT / data.AMT_ANNUITY) + data.EXT_SOURCE_2

        # interaction between rate and age
        data.loc[:, 'add_rate_age']  = (data.AMT_CREDIT / data.AMT_ANNUITY) + (-data.DAYS_BIRTH / 365)

        # interaction between age and employed and external score 2
        data.loc[:, 'add_mult_age_employed_ext_2'] = ((-data.DAYS_BIRTH / 365) +\
                                                     (-data.DAYS_EMPLOYED.replace({365243: np.nan}))) *\
                                                     (data.EXT_SOURCE_2)


        # combine ratio annuity credit, region populative relative and ext source 2
        data.loc[:, 'rate_annuity_region_ext_source_2'] = data.ratio_annuity_credit * data.REGION_POPULATION_RELATIVE * data.EXT_SOURCE_2    
        data.loc[:, 'region_ext_source_3'] = data.REGION_POPULATION_RELATIVE * data.EXT_SOURCE_3

        # Relationship between AMT_REQ_CREDIT_BUREAU_HOUR and AMT_REQ_CREDIT_BUREAU_YEAR
        data.loc[:, 'ratio_check_hour_to_year'] = data.AMT_REQ_CREDIT_BUREAU_HOUR.div(data.AMT_REQ_CREDIT_BUREAU_YEAR)

        # Relationship between Income and ratio annuity credit
        data.loc[:, 'mult_ratio_income'] = (data.ratio_annuity_credit * data.AMT_INCOME_TOTAL).map(np.log1p)
        data.loc[:, 'div_ratio_income']  = (data.AMT_INCOME_TOTAL / data.ratio_annuity_credit).map(np.log1p)

        # feature transformation
        # data = self.feature_transformation(data)

        # frequency encoding of some of the categorical variables.
        data = frequency_encoding(data, FREQ_ENCODING_COLS)

        # one hot encoding of some of the categorical variables controlled by a flag
        # if flag is True then one hot encoding else do frequency encoding.
        if compute_ohe:
            data = super(Modelv62, self).prepare_ohe(data, OHE_COLS, drop_col=True)
        else:
            data = frequency_encoding(data, OHE_COLS)
        
        return data

    # This method would perform feature engineering on merged datasets.
    def fe(self, train, test, full_test, compute_ohe=True):
        original_train = train.copy()
        data           = self.get_features(original_train, test, compute_ohe)

        train = data.iloc[:len(train)]
        test  = data.iloc[len(train):]

        del data
        gc.collect()

        if full_test is not None:
            data      = self.get_features(original_train, full_test, compute_ohe)
            full_test = data.iloc[len(train):]

            del data
            gc.collect()

        del original_train
        gc.collect()

        return train, test, full_test
    
    # OOF prediction
    def oof_preds(self, train, feature_list, model):
        X = train.loc[:, feature_list]
        y = train.loc[:, 'TARGET']

        return super(Modelv62, self).oof_preds(X, y, model)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Home Credit Default Risk')
    
    parser.add_argument('-run_name', help='Name of the experiment')
    parser.add_argument('-input_path', help='Path to input directory')
    parser.add_argument('-output_path', help='Path to output directory')
    parser.add_argument('-p', type=bool, help='Preprocess')
    parser.add_argument('-c', type=bool, help='Create Folds')
    parser.add_argument('-v', type=bool, help='Validation')
    parser.add_argument('-t', type=bool, help='Test')
    parser.add_argument('-features', type=bool, help='Generate Features')	
    parser.add_argument('-sample', type=bool, help='Work on sample')
    parser.add_argument('-seed', type=int, help='Random Seed')
    parser.add_argument('-fold_name', help='Fold Name')
    parser.add_argument('-fs', type=bool, help='Feature Selection')
    parser.add_argument('-model', type=str, help='Model Type')
    parser.add_argument('-oof', type=bool, help='OOF Preds for Test Set')
    parser.add_argument('-rank_average', type=bool, help='Rank Average OOF Preds for Test Set')    

    args    = parser.parse_args()
    
    if args.p:
        print('Preprocessing ...')
        run_name    = args.run_name
        input_path  = args.input_path
        output_path = args.output_path
        
        params = {
            'input_path': input_path,
            'output_path': output_path,
            'run_name': run_name
        }

        m  = Modelv62(**params)
        m.preprocess()

    elif args.c:
        print('Creating fold ...')

        run_name    = args.run_name
        input_path  = args.input_path
        output_path = args.output_path
        fold_name   = args.fold_name
        seed        = args.seed

        params = {
            'input_path': input_path,
            'output_path': output_path,
            'run_name': run_name
        }  

        m = Modelv62(**params)
        m.create_folds(fold_name, seed)

    elif args.features:
        print('Generating features ...')
        print()

        run_name    = args.run_name
        input_path  = args.input_path
        output_path = args.output_path
        fold_name   = args.fold_name

        params = {
            'input_path': input_path,
            'output_path': output_path,
            'run_name': run_name
        }

        m = Modelv62(**params)
        m.prepare_features(fold_name)

    elif args.t:
        print('Train Model and generate predictions on a given fold ....')
        print()

        run_name        = args.run_name
        input_path      = args.input_path
        output_path     = args.output_path
        fold_name       = args.fold_name
        is_sample       = args.sample
        model_type      = args.model
        is_oof          = args.oof

        PARAMS          = LGB_PARAMS.copy() if model_type == 'LGB' else XGB_PARAMS.copy()
        
        params = {
            'input_path'  : input_path,
            'output_path' : output_path,
            'run_name'    : run_name,
            'model_type'  : model_type
        }

        m                       = Modelv62(**params)
        train, test, full_test  = m.merge_datasets(fold_name, test_df=is_oof)
        train, test, full_test  = m.fe(train, test, full_test)

        # checking to see whether to generate a sample or not
        if is_sample is not None:
            print('Generating sample ...')
            print('Shape of training dataset before sampling: {}'.format(train.shape))
            train = m.get_sample(train, sample_size=SAMPLE_SIZE)
            print('Shape of training dataset after sampling: {}'.format(train.shape))
            print()

        # check to see if feature list exists on disk or not for a particular model
        if os.path.exists(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl')):
            feature_list = joblib.load(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl'))
        else: 
            feature_list = train.columns.tolist()
            joblib.dump(feature_list, os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl'))

        feature_list = list(set(feature_list) - set(COLS_TO_REMOVE))
        # check to see if we are doing validation or final test generation.
        is_eval  = len(fold_name) > 0
        
        if not is_eval:
            # TODO: not very fond of setting num boost round for final training this way
            # this could lead to errors if we forget to update this after doing cross-validation
            # need some other way to update it automatically.

            # TODO: also can we make SIZE_MULT (1.2) a parameter as well. 

            # use best iteration found through different folds
            PARAMS['num_boost_round'] = 2500
            PARAMS['num_boost_round'] = int(1.2 * PARAMS['num_boost_round'])
            
            if model_type == 'LGB':
                PARAMS['learning_rate']   /= 1.2
            else:
                PARAMS['eta'] /= 1.2

        # print features with null percentage
        print((train.loc[:, feature_list].isnull().sum() / len(train)).sort_values(ascending=False).iloc[:5])

        # print number of features explored in the experiment
        print('Number of features: {}'.format(len(feature_list)))

        # train model
        model, feat_df = m.train(train, test, feature_list, is_eval, **PARAMS)

        # evaluation part
        preds, score  = m.evaluate(test, feature_list, is_eval, model)
        
        # save submission
        if not is_eval:
            # save feature importance data frame to disk
            feat_df.to_csv(os.path.join(basepath, output_path + run_name + f'feat_df_{MODEL_FILENAME}.csv'), index=False)
            
            print('Generating Final Submission ...')

            # found through validation scores across multiple folds ( 3 in our current case )
            HOLDOUT_SCORE = (0.7944 + 0.7942 + 0.7898) / 3

            sub_identifier = "%s-%s-%.5f" % (datetime.now().strftime('%Y%m%d-%H%M'), MODEL_FILENAME, HOLDOUT_SCORE)

            sub            = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv.zip'))
            sub['TARGET']  = preds

            sub.to_csv(os.path.join(basepath, 'submissions/%s.csv'%(sub_identifier)), index=False)
        else:
            
            # if oof flag is True then generate predictions for test set as well
            # and save them to disk
            if is_oof:
                print('\n Generating oof predictions for test set')
                preds, _ = m.evaluate(full_test, feature_list, is_eval=False, model=model)
                joblib.dump(preds, os.path.join(basepath, output_path + run_name + f'test_preds_{MODEL_FILENAME}_{model_type}_{model.best_iteration}_{fold_name}_{score}.pkl'))

            # save feature importance data frame to disk
            feat_df.to_csv(os.path.join(basepath, output_path + run_name + f'feat_df_{MODEL_FILENAME}_{model_type}_{model.best_iteration}_{fold_name}.csv'), index=False)
            
            # only save oof predictions when it is not run in sample mode.
            if not is_sample:
                # save oof predictions
                joblib.dump(preds, os.path.join(basepath, output_path + run_name + f'preds_{MODEL_FILENAME}_{model_type}_{model.best_iteration}_{fold_name}_{score}.pkl'))
    
    elif args.fs:
        print('Feature Selection ')

        run_name        = args.run_name
        input_path      = args.input_path
        output_path     = args.output_path
        fold_name       = args.fold_name
        
        params = {
            'input_path': input_path,
            'output_path': output_path,
            'run_name': run_name
        }
        

        # Create Instance of Model Class
        m               = Modelv62(**params)
        train, test, _  = m.merge_datasets(fold_name, test_df=False)
        train, _, _     = m.fe(train, test, None)
        
        # CONSTANT FOR SAMPLE SIZE
        N_SAMPLES = int(.7 * len(train))
        train = train.sample(n=N_SAMPLES)


        # check to see if feature list exists on disk or not for a particular model
        if os.path.exists(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl')):
            feature_list = joblib.load(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl'))
        else: 
            feature_list = train.columns.tolist()
            joblib.dump(feature_list, os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl'))

        feature_list = list(set(feature_list) - set(COLS_TO_REMOVE))
         
        X = train.loc[:, feature_list]
        y = train.TARGET

        LGBM_PARAMS = {
            'colsample_bytree': .1,
            'max_depth': 6,
            'num_leaves': 54,
            'objective': 'binary',
            'n_jobs': -1
        }

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        
        cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        print('Min CV Score ( Mean ) : {}, Min CV Score ( Std ) : {}'.format(np.mean(scores), np.std(scores)))
        
        min_cv_score = np.mean(scores)
        to_remove    = []
        
        for feature in feature_list:
            print('Feature in consideration: {}'.format(feature))
            model = lgb.LGBMClassifier(**LGBM_PARAMS)

            train_copy = train.copy()
            train_copy.loc[:, feature] = train_copy.loc[:, feature].sample(frac=1).values

            fl = list(set(feature_list) - set(to_remove))
            
            X = train_copy.loc[:, fl]
            y = train_copy.TARGET

            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            print('Mean CV Score after shuffle: {}'.format(np.mean(scores)))
            
            if np.mean(scores) >= min_cv_score:
                print('Removing feature: {}'.format(feature))
                to_remove += [feature]
        
        print('\nFeatures to remove: {}'.format(to_remove))
        print('Saving to disk ...')

        filepath = os.path.join(basepath, output_path + run_name + f'{fold_name}_features_to_remove.npy')
        np.save(filepath, np.array(to_remove))


    elif args.rank_average:
        
        run_name    = args.run_name
        input_path  = args.input_path
        output_path = args.output_path
        model_type  = args.model
        

        oof_preds_filenames = [os.path.join(basepath, output_path + run_name + f'test_preds_v56_model.txt_{model_type}_5432_F0_0.7930083374342244.pkl'),
                               os.path.join(basepath, output_path + run_name + f'test_preds_v56_model.txt_{model_type}_6280_F1_0.7938754393581124.pkl'),
                               os.path.join(basepath, output_path + run_name + f'test_preds_v56_model.txt_{model_type}_5158_F2_0.7890351271935672.pkl')
                              ]

        POWER = 16

        fold0_preds = joblib.load(oof_preds_filenames[0])
        fold1_preds = joblib.load(oof_preds_filenames[1])
        fold2_preds = joblib.load(oof_preds_filenames[2])
        
        rank_average = (fold0_preds ** POWER + fold1_preds ** POWER + fold2_preds ** POWER) / 3
        
        HOLDOUT_SCORE = (0.7930 + 0.7938 + 0.7890) / 3

        print('Averaging predictions from 3-folds ....')
        sub_identifier = "%s-%s-%.5f" % (datetime.now().strftime('%Y%m%d-%H%M'), MODEL_FILENAME, HOLDOUT_SCORE)

        sub            = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv.zip'))
        sub['TARGET']  = rank_average

        sub.to_csv(os.path.join(basepath, 'submissions/%s.csv'%(sub_identifier)), index=False)
        