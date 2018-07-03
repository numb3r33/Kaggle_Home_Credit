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

basepath = os.path.expanduser('../')

SEED = 1231
np.random.seed(SEED)

#############################################################################################################
#                                       EXPERIMENT PARAMETERS                                               #                                                               
#############################################################################################################

COLS_TO_REMOVE = ['SK_ID_CURR', 
                  'TARGET',
                  'AMT_REQ_CREDIT_BUREAU_DAY',
                  'REG_REGION_NOT_LIVE_REGION',
                  'AMT_REQ_CREDIT_BUREAU_HOUR',
                  'HOUSETYPE_MODE', 
                  'FLAG_OWN_CAR', 
                  'FLAG_OWN_REALTY',
                  'NAME_EDUCATION_TYPE', 
                  'NAME_INCOME_TYPE',
                  'NAME_FAMILY_TYPE', 
                  'NAME_HOUSING_TYPE',
                  'FLAG_MOBIL',
                  'FLAG_EMP_PHONE',
                  'FLAG_WORK_PHONE',
                  'FLAG_CONT_MOBILE',
                  'FLAG_PHONE',
                  'FLAG_EMAIL',
                  'REG_REGION_NOT_LIVE_REGION',
                  'REG_REGION_NOT_WORK_REGION',
                  'LIVE_REGION_NOT_WORK_REGION',
                  'REG_CITY_NOT_LIVE_CITY',
                  'REG_CITY_NOT_WORK_CITY',
                  'LIVE_CITY_NOT_WORK_CITY',
                  'APARTMENTS_AVG',
                  'BASEMENTAREA_AVG',
                  'YEARS_BEGINEXPLUATATION_AVG',
                  'YEARS_BUILD_AVG',
                  'COMMONAREA_AVG',
                  'ELEVATORS_AVG',
                  'ENTRANCES_AVG',
                  'FLOORSMAX_AVG',
                  'FLOORSMIN_AVG',
                  'LANDAREA_AVG',
                  'LIVINGAPARTMENTS_AVG',
                  'LIVINGAREA_AVG',
                  'NONLIVINGAPARTMENTS_AVG',
                  'NONLIVINGAREA_AVG',
                  'APARTMENTS_MODE',
                  'BASEMENTAREA_MODE',
                  'YEARS_BEGINEXPLUATATION_MODE',
                  'YEARS_BUILD_MODE',
                  'COMMONAREA_MODE',
                  'ELEVATORS_MODE',
                  'ENTRANCES_MODE',
                  'FLOORSMAX_MODE',
                  'FLOORSMIN_MODE',
                  'LANDAREA_MODE',
                  'LIVINGAPARTMENTS_MODE',
                  'LIVINGAREA_MODE',
                  'NONLIVINGAPARTMENTS_MODE',
                  'NONLIVINGAREA_MODE',
                  'TOTALAREA_MODE',
                  'APARTMENTS_MEDI',
                  'BASEMENTAREA_MEDI',
                  'YEARS_BEGINEXPLUATATION_MEDI',
                  'YEARS_BUILD_MEDI',
                  'COMMONAREA_MEDI',
                  'ELEVATORS_MEDI',
                  'ENTRANCES_MEDI',
                  'FLOORSMAX_MEDI',
                  'FLOORSMIN_MEDI',
                  'LANDAREA_MEDI',
                  'LIVINGAPARTMENTS_MEDI',
                  'LIVINGAREA_MEDI',
                  'NONLIVINGAPARTMENTS_MEDI',
                  'NONLIVINGAREA_MEDI',
                  'diff_balance_curr_credit',
                  'AMT_REQ_CREDIT_BUREAU_HOUR',
                  'AMT_REQ_CREDIT_BUREAU_DAY',
                  'AMT_REQ_CREDIT_BUREAU_WEEK',
                  'AMT_REQ_CREDIT_BUREAU_MON',
                  'AMT_REQ_CREDIT_BUREAU_QRT',
                  'AMT_REQ_CREDIT_BUREAU_YEAR',
                  'ratio_min_installment_balance',
                  'HOUR_APPR_PROCESS_START',
                  'WEEKDAY_APPR_PROCESS_START',
                  'EMERGENCYSTATE_MODE',
                  'sum_num_times_prolonged',
                  'OBS_30_CNT_SOCIAL_CIRCLE',
                  'DEF_30_CNT_SOCIAL_CIRCLE',
                  'OBS_60_CNT_SOCIAL_CIRCLE',
                  'DEF_60_CNT_SOCIAL_CIRCLE',
                  'MONTHS_BALANCE_amax',
                  'MONTHS_BALANCE_median',
                  'MONTHS_BALANCE_mean',
                  'MONTHS_BALANCE_amin',
                  'MONTHS_BALANCE_var',
                  'MONTHS_BALANCE_sum',
                  'NAME_TYPE_SUITE',
                  'num_diff_credits',
                  'CNT_INSTALMENT_FUTURE_amin',
                  'NUM_NULLS_EXT_SCORES',
                  'NAME_CONTRACT_TYPE',
                  'num_high_int_no_info_loans',
                  'num_times_atm_withdrawn',
                  'change_in_credit_limit_ot',
                  'ORGANIZATION_TYPE__37',
                  'ORGANIZATION_TYPE__2',
                  'ORGANIZATION_TYPE__nan',
                  'ORGANIZATION_TYPE__32',
                  'ORGANIZATION_TYPE__52',
                  'ORGANIZATION_TYPE__48',
                  'OCCUPATION_TYPE__5',
                  'ORGANIZATION_TYPE__10',
                  'ORGANIZATION_TYPE__16',
                  'OCCUPATION_TYPE__nan',
                  'ORGANIZATION_TYPE__33',
                  'OCCUPATION_TYPE__13',
                  'ORGANIZATION_TYPE__57',
                  'ORGANIZATION_TYPE__15',
                  'ORGANIZATION_TYPE__46',
                  'ORGANIZATION_TYPE__0',
                  'ORGANIZATION_TYPE__8',
                  'ORGANIZATION_TYPE__45',
                  'ORGANIZATION_TYPE__56',
                  'ORGANIZATION_TYPE__3',
                  'ORGANIZATION_TYPE__49',
                  'ORGANIZATION_TYPE__35',
                  'ORGANIZATION_TYPE__6',
                  'OCCUPATION_TYPE__12',
                  'ORGANIZATION_TYPE__27',
                  'OCCUPATION_TYPE__15',
                  'ORGANIZATION_TYPE__22',
                  'ORGANIZATION_TYPE__25',
                  'ORGANIZATION_TYPE__23',
                  'ORGANIZATION_TYPE__9',
                  'ORGANIZATION_TYPE__21',
                  'ORGANIZATION_TYPE__18',
                  'ORGANIZATION_TYPE__44',
                  'OCCUPATION_TYPE__7',
                  'ORGANIZATION_TYPE__19',
                  'ORGANIZATION_TYPE__24',
                  'ORGANIZATION_TYPE__55',
                  'ORGANIZATION_TYPE__13',
                  'OCCUPATION_TYPE__17',
                  'ORGANIZATION_TYPE__12',
                  'ORGANIZATION_TYPE__38',
                  'ORGANIZATION_TYPE__26',
                  'ORGANIZATION_TYPE__29',
                  'ORGANIZATION_TYPE__17',
                  'ORGANIZATION_TYPE__40',
                  'OCCUPATION_TYPE__2',
                  'ORGANIZATION_TYPE__14',
                  'ORGANIZATION_TYPE__41',
                  'ORGANIZATION_TYPE__43',
                  'OCCUPATION_TYPE__16',
                  'ORGANIZATION_TYPE__1',
                  'ORGANIZATION_TYPE__50',
                  'ORGANIZATION_TYPE__53',
                  'diff_prev_credit_balance',
                  'mean_bal_payment_diff',
                  'diff_min_installment_balance',
                  'range_min_max_credit_limit',
                  'total_paid_installments',
                  'diff_bal_credit',
                  'mean_credit_limit',
                  'mean_amt_balance',
                  'OCCUPATION_TYPE__1',
                  'ORGANIZATION_TYPE__4'
                ] + [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]


LGB_PARAMS = {
    'num_boost_round': 10000,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': .01,
    'num_leaves': 16,
    'min_data_in_leaf': 100,
    'feature_fraction': .1,
    'feature_fraction_seed': SEED,
    'min_child_weight': 100,
    'lambda_l2': 20,
    'metric': 'auc',
    'nthread': 4,
    'verbose': -1,
    'seed': SEED
}


# LGB_PARAMS = {
#     'num_boost_round': 10000,
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'learning_rate': .01,
#     'metric': 'auc',
#     'max_depth': 8,
#     'min_data_in_leaf': 40,
#     'num_leaves': 44,
#     'feature_fraction': .28,
#     'feature_fraction_seed': SEED,
#     'bagging_fraction': .9,
#     'bagging_fraction_seed': SEED,
#     'lambda_l1': .3,
#     'lambda_l2': .1,
#     'min_child_weight': 98,
#     'nthread': 4,
#     'verbose': -1,
#     'seed': SEED
# }


XGB_PARAMS = {
    'num_boost_round': 10000,    
    'booster': 'gbtree',
    'silent': 1,
    'nthread': 4,
    'eta': .02,
    'max_depth': 6,
    'colsample_bytree': .6,
    'colsample_bylevel': .8,
    'objective': 'binary:logistic',
    'min_child_weight': 10,
    'eval_metric': 'auc',
    'seed': SEED
}


MODEL_FILENAME = 'v57_model.txt'
SAMPLE_SIZE    = .5

# NOTE: column in frequency encoded columns
# cannot be in ohe cols.
FREQ_ENCODING_COLS = ['ORGANIZATION_OCCUPATION',
                      'age_emp_categorical',
                      'age_occupation'
                     ]

OHE_COLS           = [
                      'ORGANIZATION_TYPE',
                      'OCCUPATION_TYPE'
                     ]



class Modelv57(BaseModel):
    def __init__(self, **params):
        self.params  = params
        self.n_train = 307511 # TODO: find a to remove this constant
    
    def load_data(self, filenames):
        dfs = []
        
        for filename in filenames:
            dfs.append(pd.read_csv(filename, parse_dates=True, keep_date_col=True))
        
        df       = pd.concat(dfs)
        df.index = np.arange(len(df))
        df       = super(Modelv57, self).reduce_mem_usage(df)

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
        dtr, dte = super(Modelv57, self).create_fold(data, seed)

        dtr.index = np.arange(len(dtr))
        dte.index = np.arange(len(dte))

        dtr.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'application_{fold_name}train.pkl'))
        dte.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'application_{fold_name}test.pkl'))

    # TODO: not super happy with the way features are constructed right now
    # because for every type of feature we are first generating those features and then persisting 
    # them to disk.
    # Possible Solution: Provide data source and feature and let a layer decide based on feature
    # type how to create and store them on the disk, that layer should be load back features given
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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

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
            data = super(Modelv57, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_pos_cash_credit_bal_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_pos_cash_credit_bal_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.time() - t0))

        else:
            print('Already generated features based on previous application, pos cash and credit card balance.')

        
        

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
                        f'prev_app_pos_cash_credit_bal_{fold_name}'                     
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
        
        return super(Modelv57, self).train_lgb(X, y, Xte, yte, **params) if self.params['model_type'] == 'LGB'\
               else super(Modelv57, self).train_xgb(X, y, Xte, yte, **params)

    # This method just takes in a model and test dataset and returns predictions 
    # prints out AUC on the test dataset as well in the process.
    def evaluate(self, test, feature_list, is_eval, model, TARGET_NAME='TARGET'):
        Xte = test.loc[:, feature_list]
        yte = []

        if is_eval:
            yte = test.loc[:, TARGET_NAME]

        return super(Modelv57, self).evaluate_lgb(Xte, yte, model) if self.params['model_type'] == 'LGB'\
               else super(Modelv57, self).evaluate_xgb(Xte, yte, model)

    # This method just takes a stratified sample of the training dataset and returns
    # back the sample.
    # TODO: add a parameter to switch between stratified and random sampling.
    def get_sample(self, train, sample_size=.5, seed=SEED):
        """
        Generates a stratified sample of provided sample_size
        """
        return super(Modelv57, self).get_sample(train, sample_size, seed=SEED)

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
        data.loc[:, 'mult_annuity_credit_ext_source_2'] = data.ratio_annuity_credit * data.EXT_SOURCE_2

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

        # frequency encoding of some of the categorical variables.
        data = frequency_encoding(data, FREQ_ENCODING_COLS)

        # one hot encoding of some of the categorical variables controlled by a flag
        # if flag is True then one hot encoding else do frequency encoding.
        if compute_ohe:
            data = super(Modelv57, self).prepare_ohe(data, OHE_COLS, drop_col=True)
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

        return super(Modelv57, self).oof_preds(X, y, model)



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

        m  = Modelv57(**params)
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

        m = Modelv57(**params)
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

        m = Modelv57(**params)
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

        m                       = Modelv57(**params)
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
            PARAMS['num_boost_round'] = 6539
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
            HOLDOUT_SCORE = (0.7933 + 0.7933 + 0.7896) / 3

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
        is_sample       = args.sample
        
        params = {
            'input_path': input_path,
            'output_path': output_path,
            'run_name': run_name
        }
        
        # Create Instance of Model Class
        m  = Modelv49(**params)
        train, test    = m.merge_datasets(fold_name)
        train, test    = m.fe(train, test)
    
        # check to see if feature list exists on disk or not for a particular model
        if os.path.exists(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl')):
            feature_list = joblib.load(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl'))
        else: 
            feature_list = train.columns.tolist()
            joblib.dump(feature_list, os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl'))

        feature_list = list(set(feature_list) - set(COLS_TO_REMOVE))
        
        # check to see if we have prepared new training dataset.
        if os.path.exists(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_new_train.pkl')):
            new_train = joblib.load(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_new_train.pkl'))
        else:

            model_params = {
                'boosting_type': 'gbdt',
                'num_leaves': 34,
                'max_depth': 6,
                'learning_rate': .02,
                'n_estimators': 2600,
                'objective': 'binary',
                'min_split_gain': .02,
                'min_child_weight': 40,
                'min_child_samples': 20,
                'subsample': .87,
                'colsample_bytree': .3,
                'reg_alpha': 0.04,
                'reg_lambda': 0.07,
                'random_state': SEED,
                'n_jobs': -1,
                'verbose': -1
            }

            lgb_model = lgb.LGBMClassifier(**model_params)
            new_train = train.copy()
            new_train.loc[:, 'prob'] = m.oof_preds(train, feature_list, lgb_model)[:, 1]

            # save dataset with probabilities to disk
            joblib.dump(new_train, os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_new_train.pkl'))

        defaulters = new_train.loc[new_train.TARGET == 1, ['TARGET', 'prob']]
        defaulters.loc[:, 'diff'] = defaulters.TARGET - defaulters.prob
        defaulters = defaulters.sort_values(by='diff', ascending=False)
        defaulters = defaulters.loc[defaulters.prob <=.5, :]

        non_defaulters = new_train.loc[new_train.TARGET == 0, ['TARGET', 'prob']]
        non_defaulters.loc[:, 'diff'] = non_defaulters.prob - non_defaulters.TARGET
        non_defaulters = non_defaulters.sort_values(by='diff', ascending=False)
        non_defaulters = non_defaulters.loc[non_defaulters.prob > .3, :]

        new_train.loc[:, 'NEW_TARGET'] = [0] * len(new_train) 
        new_train.loc[new_train.index.isin(list(defaulters.index.values) + list(non_defaulters.index.values)), 'NEW_TARGET'] = 1

        train_params = PARAMS.copy()
        train_params['num_boost_round'] = 2600
        print('Training model on new dataset ...')
        trained_model, feat_df = m.train(new_train, new_train, feature_list, is_eval=False, TARGET_NAME='NEW_TARGET', **train_params)
        
        feat_df.to_csv(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_imp_features.csv'), index=False)
    
    elif args.rank_average:
        
        run_name    = args.run_name
        input_path  = args.input_path
        output_path = args.output_path
        model_type  = args.model
        

        oof_preds_filenames = [os.path.join(basepath, output_path + run_name + f'test_preds_v56_model.txt_{model_type}_5432_F0_0.7930083374342244.pkl'),
                               os.path.join(basepath, output_path + run_name + f'test_preds_v56_model.txt_{model_type}_6280_F1_0.7938754393581124.pkl'),
                               os.path.join(basepath, output_path + run_name + f'test_preds_v56_model.txt_{model_type}_5158_F2_0.7890351271935672.pkl')
                              ]

        POWER = 4

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
        