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
                ] 

PARAMS = {
    'num_boost_round': 5000,
    'early_stopping_rounds': 100,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': .03,
    'metric': 'auc',
    'num_leaves': 35,
    'sub_feature': .1,
    'feature_fraction_seed': SEED,
    'min_data_in_leaf': 100,
    'max_bin': 300,
    'lambda_l2': 100,
    'nthread': 4,
    'verbose': -1,
    'seed': SEED
}

MODEL_FILENAME = 'v66'
SAMPLE_SIZE    = .2

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


class Modelv66(BaseModel):
    def __init__(self, **params):
        self.params  = params
        self.n_train = 307511 # TODO: find a way to remove this constant
        
    def load_data(self, filenames):
        dfs = []
        
        for filename in filenames:
            dfs.append(pd.read_csv(filename, parse_dates=True, keep_date_col=True))
        
        df       = pd.concat(dfs)
        df.index = np.arange(len(df))
        df       = super(Modelv66, self).reduce_mem_usage(df)

        return df
    
    def preprocess(self):
        
        tr = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'application_train.pkl'))
        te = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'application_test.pkl'))
        ntrain = len(tr)

        data = pd.concat((tr, te))
        
        del tr, te
        gc.collect()

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'current_application_train.pkl')):
            print('Generating features based on current application ....')

            t0 = time.clock()
            data, FEATURE_NAMES = current_application_features(data)
            data.index = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'current_application_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'current_application_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))
        else:
            print('Already generated features based on current application')


        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_train.pkl')):
            bureau = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'bureau.pkl'))
            
            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes

            print('Generating features based on credits reported to bureau ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = bureau_features(bureau, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

            del bureau
            gc.collect()

        else:
            print('Already generated features based on bureau application')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_bal_train.pkl')):
            bureau     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'bureau.pkl'))
            bureau_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'bureau_balance.pkl')) 

            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes
            
            for col in bureau_bal.select_dtypes(include=['category']).columns:
                bureau_bal.loc[:, col] = bureau_bal.loc[:, col].cat.codes

            print('Generating features based on credits reported to bureau and bureau balance ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = bureau_and_balance(bureau, bureau_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_bal_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_bal_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on bureau and balance')


        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_train.pkl')):
            prev_app = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 

            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            print('Generating features based on previous application ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_features(prev_app, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on previous application')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'pos_cash_train.pkl')):
            pos_cash = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'POS_CASH_balance.pkl')) 

            for col in pos_cash.select_dtypes(include=['category']).columns:
                pos_cash.loc[:, col] = pos_cash.loc[:, col].cat.codes
            
            print('Generating features based on pos cash ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = pos_cash_features(pos_cash, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del pos_cash
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'pos_cash_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'pos_cash_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on pos cash')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'credit_train.pkl')):
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'credit_card_balance.pkl')) 

            for col in credit_bal.select_dtypes(include=['category']).columns:
                credit_bal.loc[:, col] = credit_bal.loc[:, col].cat.codes
            
            print('Generating features based on Credit Card ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = credit_card_features(credit_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del credit_bal
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'credit_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'credit_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Credit Card')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'installments_train.pkl')):
            installments = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'installments_payments.pkl')) 

            for col in installments.select_dtypes(include=['category']).columns:
                installments.loc[:, col] = installments.loc[:, col].cat.codes
            
            print('Generating features based on Installments ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = get_installment_features(installments, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del installments
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'installments_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'installments_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Installments')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_bureau_train.pkl')):
            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            bureau     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'bureau.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes
            
            
            print('Generating features based on Previous Applications and Bureau Applications....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_bureau(prev_app, bureau, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del bureau, prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_bureau_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_bureau_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Previous application and Bureau Applications')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_credit_train.pkl')):
            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'credit_card_balance.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            for col in credit_bal.select_dtypes(include=['category']).columns:
                credit_bal.loc[:, col] = credit_bal.loc[:, col].cat.codes
            
            
            print('Generating features based on Previous Applications and Credit card balance ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_credit_card(prev_app, credit_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del credit_bal, prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_credit_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_credit_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Previous application and Credit card balance')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_installments_train.pkl')):
            prev_app     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            installments = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'installments_payments.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            for col in installments.select_dtypes(include=['category']).columns:
                installments.loc[:, col] = installments.loc[:, col].cat.codes
            
            
            print('Generating features based on Previous Applications and Installment Payments ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_installments(prev_app, installments, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del installments, prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_installments_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_installments_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Previous application and Installment Payments.')
        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'loan_stacking_train.pkl')):
            bureau     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'bureau.pkl'))
            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'credit_card_balance.pkl')) 
            
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
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'loan_stacking_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'loan_stacking_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

            del bureau
            gc.collect()
        else:
            print('Already generated features based on loan stacking.')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'feature_groups_train.pkl')):
            print('Generating features based on feature groups ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = feature_groups(data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'feature_groups_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'feature_groups_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on feature groups.')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_train.pkl')):
            print('Generating features based on previous application and pos cash ....')

            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            pos_cash   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'POS_CASH_balance.pkl')) 

            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes

            for col in pos_cash.select_dtypes(include=['category']).columns:
                pos_cash.loc[:, col] = pos_cash.loc[:, col].cat.codes
            
            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_pos(prev_app, pos_cash, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on previous application and pos cash.')


        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_credit_bal_train.pkl')):
            print('Generating features based on previous application, pos cash and credit card balance ....')

            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            pos_cash   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'POS_CASH_balance.pkl')) 
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'credit_card_balance.pkl')) 
            
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
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_credit_bal_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_credit_bal_test.pkl'))
            print('\nTook: {} seconds'.format(time.time() - t0))

        else:
            print('Already generated features based on previous application, pos cash and credit card balance.')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_ohe_train.pkl')):
            print('Generating features based on previous application one hot encoded features ....')

            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            t0                  = time.time()
            data, FEATURE_NAMES = prev_app_ohe(prev_app, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_ohe_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_ohe_test.pkl'))
            print('\nTook: {} seconds'.format(time.time() - t0))

        else:
            print('Already generated features based on previous application one hot encode features.')
    

    def prepare_features(self):
        tr = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'application_train.pkl'))
        te = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'application_test.pkl'))
        ntrain = len(tr)

        data = pd.concat((tr, te))
        
        del tr, te
        gc.collect()

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'current_application_train.pkl')):
            print('Generating features based on current application ....')

            t0 = time.clock()
            data, FEATURE_NAMES = current_application_features(data)
            data.index = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'current_application_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'current_application_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))
        else:
            print('Already generated features based on current application')


        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_train.pkl')):
            bureau = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'bureau.pkl'))
            
            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes

            print('Generating features based on credits reported to bureau ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = bureau_features(bureau, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

            del bureau
            gc.collect()

        else:
            print('Already generated features based on bureau application')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_bal_train.pkl')):
            bureau     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'bureau.pkl'))
            bureau_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'bureau_balance.pkl')) 

            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes
            
            for col in bureau_bal.select_dtypes(include=['category']).columns:
                bureau_bal.loc[:, col] = bureau_bal.loc[:, col].cat.codes

            print('Generating features based on credits reported to bureau and bureau balance ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = bureau_and_balance(bureau, bureau_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_bal_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'bureau_bal_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on bureau and balance')


        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_train.pkl')):
            prev_app = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 

            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            print('Generating features based on previous application ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_features(prev_app, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on previous application')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'pos_cash_train.pkl')):
            pos_cash = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'POS_CASH_balance.pkl')) 

            for col in pos_cash.select_dtypes(include=['category']).columns:
                pos_cash.loc[:, col] = pos_cash.loc[:, col].cat.codes
            
            print('Generating features based on pos cash ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = pos_cash_features(pos_cash, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del pos_cash
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'pos_cash_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'pos_cash_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on pos cash')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'credit_train.pkl')):
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'credit_card_balance.pkl')) 

            for col in credit_bal.select_dtypes(include=['category']).columns:
                credit_bal.loc[:, col] = credit_bal.loc[:, col].cat.codes
            
            print('Generating features based on Credit Card ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = credit_card_features(credit_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del credit_bal
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'credit_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'credit_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Credit Card')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'installments_train.pkl')):
            installments = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'installments_payments.pkl')) 

            for col in installments.select_dtypes(include=['category']).columns:
                installments.loc[:, col] = installments.loc[:, col].cat.codes
            
            print('Generating features based on Installments ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = get_installment_features(installments, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del installments
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'installments_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'installments_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Installments')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_bureau_train.pkl')):
            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            bureau     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'bureau.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes
            
            
            print('Generating features based on Previous Applications and Bureau Applications....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_bureau(prev_app, bureau, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del bureau, prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_bureau_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_bureau_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Previous application and Bureau Applications')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_credit_train.pkl')):
            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'credit_card_balance.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            for col in credit_bal.select_dtypes(include=['category']).columns:
                credit_bal.loc[:, col] = credit_bal.loc[:, col].cat.codes
            
            
            print('Generating features based on Previous Applications and Credit card balance ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_credit_card(prev_app, credit_bal, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del credit_bal, prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_credit_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_credit_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Previous application and Credit card balance')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_installments_train.pkl')):
            prev_app     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            installments = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'installments_payments.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            for col in installments.select_dtypes(include=['category']).columns:
                installments.loc[:, col] = installments.loc[:, col].cat.codes
            
            
            print('Generating features based on Previous Applications and Installment Payments ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_installments(prev_app, installments, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            del installments, prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_installments_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_installments_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Previous application and Installment Payments.')
        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'loan_stacking_train.pkl')):
            bureau     = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'bureau.pkl'))
            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'credit_card_balance.pkl')) 
            
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
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'loan_stacking_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'loan_stacking_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

            del bureau
            gc.collect()
        else:
            print('Already generated features based on loan stacking.')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'feature_groups_train.pkl')):
            print('Generating features based on feature groups ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = feature_groups(data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'feature_groups_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'feature_groups_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on feature groups.')

        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_train.pkl')):
            print('Generating features based on previous application and pos cash ....')

            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            pos_cash   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'POS_CASH_balance.pkl')) 

            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes

            for col in pos_cash.select_dtypes(include=['category']).columns:
                pos_cash.loc[:, col] = pos_cash.loc[:, col].cat.codes
            
            t0                  = time.clock()
            data, FEATURE_NAMES = prev_app_pos(prev_app, pos_cash, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on previous application and pos cash.')


        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_credit_bal_train.pkl')):
            print('Generating features based on previous application, pos cash and credit card balance ....')

            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            pos_cash   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'POS_CASH_balance.pkl')) 
            credit_bal = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'credit_card_balance.pkl')) 
            
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
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_credit_bal_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_pos_cash_credit_bal_test.pkl'))
            print('\nTook: {} seconds'.format(time.time() - t0))

        else:
            print('Already generated features based on previous application, pos cash and credit card balance.')

        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_ohe_train.pkl')):
            print('Generating features based on previous application one hot encoded features ....')

            prev_app   = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + 'previous_application.pkl')) 
            
            for col in prev_app.select_dtypes(include=['category']).columns:
                prev_app.loc[:, col] = prev_app.loc[:, col].cat.codes
            
            t0                  = time.time()
            data, FEATURE_NAMES = prev_app_ohe(prev_app, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(Modelv66, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_ohe_train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'prev_app_ohe_test.pkl'))
            print('\nTook: {} seconds'.format(time.time() - t0))

        else:
            print('Already generated features based on previous application one hot encode features.')
    
    # This method currently takes care of loading engineered features from disk
    # and merging train and test to report back a dataframe (data) which can be used by
    # other layers.
    def merge_datasets(self):

        def get_filenames():
            filenames = [f'application_',
                        f'current_application_',
                        f'bureau_',
                        f'bureau_bal_',
                        f'prev_app_',
                        f'pos_cash_',
                        f'credit_',					 
                        f'installments_',					 
                        f'prev_app_bureau_',					 
                        f'prev_app_credit_',				 
                        f'prev_app_installments_',
                        f'loan_stacking_',
                        f'feature_groups_',
                        f'prev_app_pos_cash_',
                        f'prev_app_pos_cash_credit_bal_',
                        f'prev_app_ohe_'                                                      
                        ]

            return filenames

        train     = []
        test      = []
        
        filenames = get_filenames()
        for filename_ in filenames:
            tmp       = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'{filename_}train.pkl'))
            tmp.index = np.arange(len(tmp)) 
            train.append(tmp)

        for filename_ in filenames:
            tmp       = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + 'feature_groups/' + f'{filename_}test.pkl'))
            tmp.index = np.arange(len(tmp))
            test.append(tmp)

        return pd.concat(train, axis=1), pd.concat(test, axis=1)
    
    def feature_interaction(self, data, key, agg_feature, agg_func, agg_func_name):
        key_name = '_'.join(key)
        
        tmp = data.groupby(key)[agg_feature].apply(agg_func)\
                .reset_index()\
                .rename(columns={agg_feature: f'{agg_func_name}_{key_name}_{agg_feature}'})
        
        data.loc[:, f'{agg_func_name}_{key_name}_{agg_feature}'] = data.loc[:, key]\
                                                            .merge(tmp, on=key, how='left')[f'{agg_func_name}_{key_name}_{agg_feature}']
        
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

        # Gender, Education and Ext scores
        data = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_2', np.var, 'var')
        
        # Gender, Occupation and Ext scores    
        data = self.feature_interaction(data, ['CODE_GENDER', 'OCCUPATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        
        # Gender, Organization and Ext score
        data = self.feature_interaction(data, ['CODE_GENDER', 'ORGANIZATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        
        # Income, Occupation and Ext Score
        data = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        
        # Occupation and Organization and Ext Score
        data = self.feature_interaction(data, ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        
        # Income, Education and Ext score
        data = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        
        # feature transformation
        # data = self.feature_transformation(data)

        # frequency encoding of some of the categorical variables.
        data = frequency_encoding(data, FREQ_ENCODING_COLS)

        # one hot encoding of some of the categorical variables controlled by a flag
        # if flag is True then one hot encoding else do frequency encoding.
        if compute_ohe:
            data = super(Modelv66, self).prepare_ohe(data, OHE_COLS, drop_col=True)
        else:
            data = frequency_encoding(data, OHE_COLS)
        
        return data

    # This method would perform feature engineering on merged datasets.
    def fe(self, train, test, compute_ohe=True):
        original_train = train.copy()
        data           = self.get_features(original_train, test, compute_ohe)

        train = data.iloc[:len(train)]
        test  = data.iloc[len(train):]

        del data, original_train
        gc.collect()

        return train, test

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
        
        return super(Modelv66, self).train_lgb(X, y, Xte, yte, **params)

    # This method just takes in a model and test dataset and returns predictions 
    # prints out AUC on the test dataset as well in the process.
    def evaluate(self, test, feature_list, is_eval, model, TARGET_NAME='TARGET'):
        Xte = test.loc[:, feature_list]
        yte = []

        if is_eval:
            yte = test.loc[:, TARGET_NAME]

        return super(Modelv66, self).evaluate_lgb(Xte, yte, model)

    def cross_validate(self, train, feature_list, params, TARGET_NAME='TARGET'):
        Xtr = train.loc[:, feature_list]
        ytr = train.loc[:, TARGET_NAME]

        return super(Modelv66, self).cross_validate(Xtr, ytr, params)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Home Credit Default Risk Solution')
    
    parser.add_argument('-input_path', help='Path to input directory')     # path to raw files
    parser.add_argument('-output_path', help='Path to output directory')   # path to working data folder 
    parser.add_argument('-data_folder', help='Folder name of the dataset') # dataset folder name
    parser.add_argument('-p', type=bool, help='Preprocess')
    parser.add_argument('-cv', type=bool, help='Cross Validation')
    parser.add_argument('-v', type=str, help='Validation')
    parser.add_argument('-features', type=bool, help='Generate Features')
    parser.add_argument('-s', type=bool, help='Whether to work on a sample or not.')
    parser.add_argument('-seed', type=int, help='Random SEED')
    parser.add_argument('-t', type=bool, help='Full Training Loop.')
    parser.add_argument('-ensemble', type=bool , help='Average out predictions.')

    args    = parser.parse_args()

    if args.p:
        print('Preprocessing ...')
        input_path  = args.input_path
        output_path = args.output_path
        
        params = {
            'input_path': input_path,
            'output_path': output_path
        }

        m  = Modelv66(**params)
        m.preprocess()

    elif args.features:
        print('Generating features ...')
        print()

        input_path  = args.input_path
        output_path = args.output_path
        
        params = {
            'input_path': input_path,
            'output_path': output_path,
        }

        m = Modelv66(**params)
        m.prepare_features()

    elif args.v is not None and len(args.v):
        print('Train and generate predictions on a fold')

        input_path      = args.input_path
        output_path     = args.output_path
        data_folder     = args.data_folder
        fold_indicator  = args.v
        is_sample       = args.s
        SEED            = int(args.seed)

        print('*' * 100)
        print('SEED FOUND: {}'.format(SEED))

        params = {
            'input_path': input_path,
            'output_path': output_path
        }

        # Set seed to Params
        PARAMS['seed'] = SEED
        PARAMS['feature_fraction_seed'] = SEED
        PARAMS['bagging_seed'] = SEED

        m   = Modelv66(**params)
            
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}data.h5')):
            print('Loading dataset from disk ...')
            data = pd.read_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')
        else:
            print('Merge feature groups and save them to disk ...')
            train, test  = m.merge_datasets()
            train, test  = m.fe(train, test)
            
            data         = pd.concat((train, test))
            data.to_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')

            del train, test
            gc.collect()

        itr    = pd.read_csv(os.path.join(basepath, input_path + 'cv_idx.csv'), usecols=[fold_indicator])[fold_indicator].values
        print('Shape of fold indices ', len(itr))

        ite    = np.array(list(set(np.arange(m.n_train)) - set(itr)))
        
        train  = data.iloc[:m.n_train].iloc[itr]
        test   = data.iloc[:m.n_train].iloc[ite]

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

        
        # print features with null percentage
        print('Top-5 features with highest percentage of null values ...\n')
        print((train.loc[:, feature_list].isnull().sum() / len(train)).sort_values(ascending=False).iloc[:5])
        
        # print number of features explored in the experiment
        print('*' * 100)
        print('Number of features: {}'.format(len(feature_list)))

        print('*' * 100)

        model_identifier = f'{data_folder}{MODEL_FILENAME}_{fold_indicator}_{SEED}'

        if os.path.exists(os.path.join(basepath, output_path + f'{model_identifier}_model.txt')):
            print('Loading model from disk ...')
            model = lgb.Booster(model_file=os.path.join(basepath, output_path + f'{model_identifier}_model.txt'))
            yhold = test.TARGET
            hold_preds = np.array(model.predict(test.loc[:, feature_list]))
            print('AUC score: {}'.format(roc_auc_score(yhold, hold_preds)))

        else:
            print('Saving model to disk ...')
            # train model
            model, feat_df = m.train(train, test, feature_list, is_eval=True, **PARAMS)
            
            if not is_sample:
                model.save_model(os.path.join(basepath, output_path + f'{model_identifier}_model.txt'))
                
                if not os.path.exists(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_true_holdout.npy')):
                    np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_true_holdout.npy'), test.TARGET)
                
                hold_preds = model.predict(test.loc[:, feature_list])
                np.save(os.path.join(basepath, output_path + f'{model_identifier}_preds_holdout.npy'), hold_preds)
                feat_df.to_csv(os.path.join(basepath, output_path + f'{model_identifier}_feat_imp.csv'), index=False)
    
    elif args.cv:
        print('Cross validation on training and store parameters and cv score on disk ...')

        input_path      = args.input_path
        output_path     = args.output_path
        data_folder     = args.data_folder
        is_sample       = args.s

        params = {
            'input_path': input_path,
            'output_path': output_path
        }

        m   = Modelv66(**params)
            
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}data.h5')):
            print('Loading dataset from disk ...')
            data = pd.read_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')
        else:
            print('Merge feature groups and save them to disk ...')
            train, test  = m.merge_datasets()
            train, test  = m.fe(train, test)
            
            data         = pd.concat((train, test))
            data.to_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')

            del train, test
            gc.collect()

        train  = data.iloc[:m.n_train]
        
        if is_sample:
            print('*' * 100)
            print('Take a random sample of the training data ...')
            train = train.sample(frac=SAMPLE_SIZE)
        
        # check to see if feature list exists on disk or not for a particular model
        if os.path.exists(os.path.join(basepath, output_path + 'chosen_features.npy')):
            feature_list = np.load(os.path.join(basepath, output_path + 'chosen_features.npy'))
        else: 
            feature_list = train.columns.tolist()
            feature_list = list(set(feature_list) - set(COLS_TO_REMOVE))
            np.save(os.path.join(basepath, output_path + 'chosen_features.npy'), feature_list)

        cv_history = m.cross_validate(train, feature_list, PARAMS.copy())
        cv_score   = str(cv_history.iloc[-1]['auc-mean']) + '_' + str(cv_history.iloc[-1]['auc-stdv'])
        
        PARAMS['num_boost_round'] = len(cv_history)

        joblib.dump(PARAMS, os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_params.pkl'))
        joblib.dump(cv_score, os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_cv.pkl'))
    
    elif args.t:
        print('Full Training')

        input_path      = args.input_path
        output_path     = args.output_path
        data_folder     = args.data_folder
        SEED            = int(args.seed)
        
        params = {
            'input_path': input_path,
            'output_path': output_path
        }

        m   = Modelv66(**params)
        
        # Load or save data from/ on disk
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}data.h5')):
            print('Loading dataset from disk ...')
            data = pd.read_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')
        else:
            print('Merge feature groups and save them to disk ...')
            train, test  = m.merge_datasets()
            train, test  = m.fe(train, test)
            
            data         = pd.concat((train, test))
            data.to_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')

            del train, test
            gc.collect()

        # separate out training and test set.
        train  = data.iloc[:m.n_train]
        test   = data.iloc[m.n_train:]

        # check to see if feature list exists on disk or not for a particular model
        if os.path.exists(os.path.join(basepath, output_path + 'chosen_features.npy')):
            feature_list = np.load(os.path.join(basepath, output_path + 'chosen_features.npy'))
        else: 
            feature_list = train.columns.tolist()
            feature_list = list(set(feature_list) - set(COLS_TO_REMOVE))
            np.save(os.path.join(basepath, output_path + 'chosen_features.npy'), feature_list)

        
        # Load params and holdout score from disk.
        PARAMS        = joblib.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_params.pkl'))
        HOLDOUT_SCORE = joblib.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_cv.pkl'))

        PARAMS['num_boost_round'] = int(1.2 * PARAMS['num_boost_round'])
        PARAMS['learning_rate']  /= 1.2

        PARAMS['seed'] = SEED
        PARAMS['feature_fraction_seed'] = SEED
        PARAMS['bagging_seed'] = SEED

        print('*' * 100)
        print('PARAMS are: {}'.format(PARAMS))

        # train model
        model, feat_df = m.train(train, test, feature_list, is_eval=False, **PARAMS)
        
        # evaluation part
        preds, score  = m.evaluate(test, feature_list, is_eval=False, model=model)

        sub_identifier = "%s-%s-%s-%s" % (datetime.now().strftime('%Y%m%d-%H%M'), MODEL_FILENAME, HOLDOUT_SCORE, SEED)

        sub            = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv.zip'))
        sub['TARGET']  = preds

        sub.to_csv(os.path.join(basepath, 'submissions/%s.csv'%(sub_identifier)), index=False)
    
    elif args.ensemble:

        output_files = [os.path.join(basepath, 'submissions/20180714-0003-v66-0.7955457637350173_0.002634437556483681-1231.csv'),
                        os.path.join(basepath, 'submissions/20180714-0011-v66-0.7955457637350173_0.002634437556483681-2001.csv'),
                        os.path.join(basepath, 'submissions/20180714-0017-v66-0.7955457637350173_0.002634437556483681-2193.csv')                        
                       ]

        ensemble_preds = 0

        for f in output_files:
            sub = pd.read_csv(f)['TARGET'].values
            ensemble_preds += sub
        
        ensemble_preds /= len(output_files)
        HOLDOUT_SCORE   = .79479
    
        sub_identifier = "%s-%s-%s" % (datetime.now().strftime('%Y%m%d-%H%M'), MODEL_FILENAME, HOLDOUT_SCORE)
        sub            = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv.zip'))
        sub['TARGET']  = ensemble_preds

        sub.to_csv(os.path.join(basepath, 'submissions/ensemble_%s.csv'%(sub_identifier)), index=False)
    