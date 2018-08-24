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

COLS_TO_REMOVE = ['TARGET',
                  'mean_NAME_INCOME_TYPE_AMT_CREDIT',
                  'ORGANIZATION_TYPE__42',
                  'NAME_YIELD_GROUP_2',
                  'FLAG_OWN_REALTY',
                  'AMT_REQ_CREDIT_BUREAU_MON',
                  'recent_employment',
                  'STATUS5.0',
                  'LIVE_CITY_NOT_WORK_CITY',
                  'OCCUPATION_TYPE__8',
                  'ORGANIZATION_TYPE__5',
                  'PRODUCT_COMBINATION_10',
                  'STATUS4.0',
                  'STATUS3.0',
                  'NAME_HOUSING_TYPE__1',
                  'mean_NAME_INCOME_TYPE_EXT_SOURCE_2',
                  'FLAG_PHONE',
                  'REG_CITY_NOT_WORK_CITY',
                  'HOUR_APPR_PROCESS_START_4',
                  'NAME_GOODS_CATEGORY_6',
                  'PRODUCT_COMBINATION_15',
                  'cnt_non_child',
                  'MONTHS_BALANCE_amax',
                  'NAME_HOUSING_TYPE__5',
                  'mean_NAME_INCOME_TYPE_EXT_SOURCE_3',
                  'ORGANIZATION_TYPE__7',
                  'NAME_GOODS_CATEGORY_3',
                  'NAME_CONTRACT_STATUS_3',
                  'CODE_REJECT_REASON_0',
                  'NAME_GOODS_CATEGORY_22',
                  'OCCUPATION_TYPE__4',
                  'HOUR_APPR_PROCESS_START_19',
                  'CODE_REJECT_REASON_8',
                  'NAME_HOUSING_TYPE__4',
                  'NAME_GOODS_CATEGORY_4',
                  'NAME_GOODS_CATEGORY_11',
                  'AMT_REQ_CREDIT_BUREAU_WEEK',
                  'ORGANIZATION_TYPE__54',
                  'sum_num_times_prolonged',
                  'PRODUCT_COMBINATION_14',
                  'NAME_INCOME_TYPE__1',
                  'HOUR_APPR_PROCESS_START_3',
                  'CODE_REJECT_REASON_6',
                  'ORGANIZATION_TYPE__55',
                  'HOUR_APPR_PROCESS_START_20',
                  'mean_NAME_EDUCATION_TYPE_EXT_SOURCE_2',
                  'OCCUPATION_TYPE__14',
                  'ORGANIZATION_TYPE__51',
                  'NAME_HOUSING_TYPE__2',
                  'OCCUPATION_TYPE__16',
                  'AMT_REQ_CREDIT_BUREAU_DAY',
                  'NAME_CONTRACT_TYPE',
                  'REG_REGION_NOT_LIVE_REGION',
                  'sum_CODE_GENDER_NAME_EDUCATION_TYPE_OWN_CAR_AGE',
                  'NAME_GOODS_CATEGORY_20',
                  'OCCUPATION_TYPE__9',
                  'REG_REGION_NOT_WORK_REGION',
                  'HOUR_APPR_PROCESS_START_21',
                  'NAME_GOODS_CATEGORY_17',
                  'mean_NAME_EDUCATION_TYPE_EXT_SOURCE_3',
                  'ORGANIZATION_TYPE__47',
                  'FLAG_DOCUMENT_8',
                  'OCCUPATION_TYPE__3',
                  'OCCUPATION_TYPE__2',
                  'LIVE_REGION_NOT_WORK_REGION',
                  'ORGANIZATION_TYPE__40',
                  'ORGANIZATION_TYPE__4',
                  'FLAG_EMAIL',
                  'ORGANIZATION_TYPE__38',
                  'FLAG_DOCUMENT_5',
                  'NAME_GOODS_CATEGORY_13',
                  'ORGANIZATION_TYPE__33',
                  'ratio_sum_C_X_to_missing',
                  'NAME_GOODS_CATEGORY_16',
                  'HOUR_APPR_PROCESS_START_2',
                  'ORGANIZATION_TYPE__11',
                  'ORGANIZATION_TYPE__20',
                  'PRODUCT_COMBINATION_16',
                  'ORGANIZATION_TYPE__3',
                  'NAME_EDUCATION_TYPE__3',
                  'FLAG_DOCUMENT_6',
                  'OCCUPATION_TYPE__-1',
                  'ORGANIZATION_TYPE__30',
                  'ORGANIZATION_TYPE__1',
                  'ratio_6_12_prev_app_months',
                  'CHANNEL_TYPE_2',
                  'ORGANIZATION_TYPE__28',
                  'ORGANIZATION_TYPE__39',
                  'NAME_GOODS_CATEGORY_25',
                  'NAME_GOODS_CATEGORY_8',
                  'ORGANIZATION_TYPE__43',
                  'ORGANIZATION_TYPE__36',
                  'PRODUCT_COMBINATION_12',
                  'ORGANIZATION_TYPE__21',
                  'OCCUPATION_TYPE__11',
                  'CHANNEL_TYPE_1',
                  'NAME_HOUSING_TYPE__0',
                  'ratio_check_hour_to_year',
                  'NAME_GOODS_CATEGORY_18',
                  'OCCUPATION_TYPE__12',
                  'OCCUPATION_TYPE__17',
                  'NAME_GOODS_CATEGORY_15',
                  'NAME_INCOME_TYPE__4',
                  'NAME_INCOME_TYPE__3',
                  'NAME_GOODS_CATEGORY_12',
                  'ORGANIZATION_TYPE__53',
                  'FLAG_EMP_PHONE',
                  'CODE_REJECT_REASON_5',
                  'ORGANIZATION_TYPE__13',
                  'OCCUPATION_TYPE__10',
                  'FLAG_DOCUMENT_9',
                  'OCCUPATION_TYPE__6',
                  'ORGANIZATION_TYPE__14',
                  'ORGANIZATION_TYPE__16',
                  'OCCUPATION_TYPE__1',
                  'NAME_CONTRACT_TYPE3',
                  'NAME_GOODS_CATEGORY_24',
                  'ORGANIZATION_TYPE__12',
                  'OCCUPATION_TYPE__13',
                  'FLAG_DOCUMENT_18',
                  'HOUR_APPR_PROCESS_START_23',
                  'PRODUCT_COMBINATION_-1',
                  'NAME_EDUCATION_TYPE__2',
                  'ORGANIZATION_TYPE__10',
                  'ORGANIZATION_TYPE__24',
                  'ORGANIZATION_TYPE__45',
                  'NAME_GOODS_CATEGORY_23',
                  'OCCUPATION_TYPE__7',
                  'AMT_REQ_CREDIT_BUREAU_HOUR',
                  'ORGANIZATION_TYPE__44',
                  'ORGANIZATION_TYPE__22',
                  'ORGANIZATION_TYPE__35',
                  'NAME_GOODS_CATEGORY_0',
                  'OCCUPATION_TYPE__15',
                  'NAME_GOODS_CATEGORY_21',
                  'ORGANIZATION_TYPE__32',
                  'HOUR_APPR_PROCESS_START_22',
                  'HOUR_APPR_PROCESS_START_1',
                  'ORGANIZATION_TYPE__0',
                  'FLAG_CONT_MOBILE',
                  'ORGANIZATION_TYPE__26',
                  'ORGANIZATION_TYPE__2',
                  'OCCUPATION_TYPE__0',
                  'ORGANIZATION_TYPE__41',
                  'HOUR_APPR_PROCESS_START_0',
                  'ORGANIZATION_TYPE__6',
                  'contract_status_str',
                  'NAME_HOUSING_TYPE__3',
                  'ORGANIZATION_TYPE__34',
                  'ORGANIZATION_TYPE__56',
                  'ORGANIZATION_TYPE__31',
                  'FLAG_DOCUMENT_14',
                  'OCCUPATION_TYPE__5',
                  'ORGANIZATION_TYPE__29',
                  'NAME_INCOME_TYPE__6',
                  'FLAG_DOCUMENT_7',
                  'ORGANIZATION_TYPE__57',
                  'ORGANIZATION_TYPE__27',
                  'ORGANIZATION_TYPE__46',
                  'ORGANIZATION_TYPE__8',
                  'FLAG_DOCUMENT_16',
                  'NAME_GOODS_CATEGORY_26',
                  'NAME_GOODS_CATEGORY_1',
                  'FLAG_DOCUMENT_19',
                  'FLAG_DOCUMENT_20',
                  'FLAG_DOCUMENT_11',
                  'ORGANIZATION_TYPE__19',
                  'NAME_GOODS_CATEGORY_14',
                  'ORGANIZATION_TYPE__52',
                  'NAME_GOODS_CATEGORY_9',
                  'ORGANIZATION_TYPE__9',
                  'FLAG_DOCUMENT_13',
                  'ORGANIZATION_TYPE__50',
                  'NAME_GOODS_CATEGORY_10',
                  'ORGANIZATION_TYPE__17',
                  'ORGANIZATION_TYPE__48',
                  'ORGANIZATION_TYPE__37',
                  'FLAG_DOCUMENT_15',
                  'ORGANIZATION_TYPE__23',
                  'FLAG_DOCUMENT_21',
                  'ORGANIZATION_TYPE__15',
                  'FLAG_DOCUMENT_2',
                  'mean_CODE_GENDER_ORGANIZATION_TYPE_DAYS_REGISTRATION',
                  'NAME_INCOME_TYPE__5',
                  'FLAG_DOCUMENT_12',
                  'diff_code_gender_organization_type_days_reg_mean',
                  'ORGANIZATION_TYPE__25',
                  'NAME_INCOME_TYPE__2',
                  'FLAG_DOCUMENT_17',
                  'FLAG_MOBIL',
                  'NAME_EDUCATION_TYPE__nan',
                  'ORGANIZATION_TYPE__18',
                  'ORGANIZATION_TYPE__nan',
                  'FLAG_DOCUMENT_10',
                  'ORGANIZATION_TYPE__49',
                  'NAME_INCOME_TYPE__0',
                  'NAME_EDUCATION_TYPE__0',
                  'FLAG_DOCUMENT_4',
                  'diff_max_min_credit_term',
                  'NAME_HOUSING_TYPE__nan',
                  'OCCUPATION_TYPE__nan',
                  'NAME_INCOME_TYPE__nan',
                  'home_credit_years_cat'
                  ]  

PARAMS = {
    'n_estimators': 1000,
    'max_depth': 12,
    'max_features': 'sqrt',
    'n_jobs': -1,
    'random_state': SEED
}


PCA_PARAMS = {
    'n_components': 10,
    'whiten': True,
    'random_state': SEED
}


MODEL_FILENAME           = 'v132'
SAMPLE_SIZE              = .3

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

TARGET_ENCODING_COLS = []


class Modelv132(BaseModel):
    def __init__(self, **params):
        self.params  = params
        self.n_train = 307511 # TODO: find a way to remove this constant
        
    def load_data(self, filenames):
        dfs = []
        
        for filename in filenames:
            dfs.append(pd.read_csv(filename, parse_dates=True, keep_date_col=True))
        
        df       = pd.concat(dfs)
        df.index = np.arange(len(df))
        df       = super(Modelv132, self).reduce_mem_usage(df)

        return df
    
    def reduce_mem_usage(self, df):
        return super(Modelv132, self).reduce_mem_usage(df)
    
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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
            data = super(Modelv132, self).fill_infrequent_values(data)

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
        
        feat_name = f'{agg_func_name}_{key_name}_{agg_feature}'
        data.loc[:, feat_name] = data.loc[:, key].merge(tmp, on=key, how='left')[feat_name]
        
        return data, feat_name

    def feature_preprocessing(self, data):
        # current application preprocessing
        data['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        data['CODE_GENDER'].replace(2, np.nan, inplace=True)
        data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

        # previous application
        data['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        data['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        data['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        data['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        data['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
        
        return data

    def add_missing_values_flag(self, data):
        # preprocess for pca
        SKIP_COLS = ['SK_ID_CURR', 'TARGET']
        
        for col in data.columns.drop(SKIP_COLS):
            # replace inf with np.nan
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            
            # fill missing values with median
            if data[col].isnull().sum():
                data[f'{col}_flag'] = data[col].isnull().astype(np.uint8)

                if pd.isnull(data[col].median()):
                    data[col] = data[col].fillna(-1)
                else:
                    data[col] = data[col].fillna(data[col].median())

        return data
    
    def get_features(self, train, test, compute_categorical):
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

        # Gender, Education and other features
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_name_education_type_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_2', np.var, 'var')
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_name_education_type_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']
                
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_name_education_type_amt_credit_mean'] = data[feat_name] - data['AMT_CREDIT']
        
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_name_education_type_amt_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']
        
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'OWN_CAR_AGE', np.max, 'max')
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'OWN_CAR_AGE', np.sum, 'sum')

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_education_type_age'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_education_type_empl'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_education_type_income'] = data[feat_name] - data['AMT_INCOME_TOTAL']


        # Gender, Occupation and other features    
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'OCCUPATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_occupation_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'OCCUPATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_occupation_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']
        
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'OCCUPATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_occupation_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'OCCUPATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_occupation_days_birth_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'OCCUPATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_occupation_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'OCCUPATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_occupation_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'OCCUPATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_occupation_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'OCCUPATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_occupation_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']
        
        # Gender, Organization and other features
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'ORGANIZATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_organization_type_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']
        
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'ORGANIZATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_organization_type_amt_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']
        
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'ORGANIZATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_organization_type_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']
        
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'ORGANIZATION_TYPE'], 'DAYS_REGISTRATION', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_organization_type_days_reg_mean'] = data[feat_name] - data['DAYS_REGISTRATION']
        
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'ORGANIZATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_organization_type_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'ORGANIZATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_organization_type_age_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'ORGANIZATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_organization_type_empl_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'ORGANIZATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_organization_type_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']
        
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'ORGANIZATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_organization_type_credit_mean'] = data[feat_name] - data['AMT_CREDIT']
        
        # Gender, Reg city not work city and other fatures
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_reg_city_amount_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']
        
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], 'CNT_CHILDREN', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_reg_city_cnt_children_mean'] = data[feat_name] - data['CNT_CHILDREN']
        
        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], 'DAYS_ID_PUBLISH', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_reg_city_days_id_mean'] = data[feat_name] - data['DAYS_ID_PUBLISH']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_reg_city_age_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_reg_city_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_reg_city_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_code_gender_reg_city_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']


        # Income, Occupation and Ext Score
        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_name_income_type_occupation_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_name_income_type_occupation_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_name_income_type_occupation_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']


        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_name_income_type_occupation_age_mean'] = data[feat_name] - data['DAYS_BIRTH']
        
        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_name_income_type_occupation_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_name_income_type_occupation_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_name_income_type_occupation_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_name_income_type_occupation_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']

        
        # Occupation and Organization and Ext Score
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_occupation_organization_source_2_mean'] = data[feat_name] - data['DAYS_ID_PUBLISH']

        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_occupation_organization_source_1_mean'] = data[feat_name] - data['DAYS_ID_PUBLISH']

        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_occupation_organization_source_3_mean'] = data[feat_name] - data['DAYS_ID_PUBLISH']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_occupation_organization_age_mean'] = data[feat_name] - data['DAYS_BIRTH']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_occupation_organization_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_occupation_organization_credit_mean'] = data[feat_name] - data['AMT_CREDIT']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_occupation_organization_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_occupation_organization_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']


        # Income, Education and Ext score
        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_income_type_education_type_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_income_type_education_type_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_income_type_education_type_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_income_type_education_type_age_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_income_type_education_type_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_income_type_education_type_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_income_type_education_type_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_income_type_education_type_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']


        # Education and Occupation and other features
        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_education_occupation_amt_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_education_occupation_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_education_occupation_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_education_occupation_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], 'OWN_CAR_AGE', np.mean, 'mean')        
        data.loc[:, 'diff_education_occupation_car_age_mean'] = data[feat_name] - data['OWN_CAR_AGE']
        
        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')        
        data.loc[:, 'diff_education_occupation_age_mean'] = data[feat_name] - data['DAYS_BIRTH']
        
        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')        
        data.loc[:, 'diff_education_occupation_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']
        
        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')        
        data.loc[:, 'diff_education_occupation_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']
        
        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')        
        data.loc[:, 'diff_education_occupation_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']
        

        # Education, Occupation, Reg city not work city and other features
        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], 'EXT_SOURCE_2', np.mean, 'mean')         
        data.loc[:, 'diff_education_occupation_ext_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], 'EXT_SOURCE_1', np.mean, 'mean')         
        data.loc[:, 'diff_education_occupation_ext_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], 'EXT_SOURCE_3', np.mean, 'mean')         
        data.loc[:, 'diff_education_occupation_ext_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']


        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], 'DAYS_BIRTH', np.mean, 'mean')         
        data.loc[:, 'diff_education_occupation_age_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], 'DAYS_EMPLOYED', np.mean, 'mean')         
        data.loc[:, 'diff_education_occupation_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], 'AMT_CREDIT', np.mean, 'mean')         
        data.loc[:, 'diff_education_occupation_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], 'AMT_ANNUITY', np.mean, 'mean')         
        data.loc[:, 'diff_education_occupation_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']
        
        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], 'AMT_INCOME_TOTAL', np.mean, 'mean')         
        data.loc[:, 'diff_education_occupation_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']
        

        # Occupation and other features
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean') 
        data.loc[:, 'diff_occupation_reg_city_mean'] = data[feat_name] - data['AMT_ANNUITY']
                
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'CNT_CHILDREN', np.mean, 'mean')         
        data.loc[:, 'diff_occupation_cnt_children_mean'] = data[feat_name] - data['CNT_CHILDREN']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'CNT_FAM_MEMBERS', np.mean, 'mean')         
        data.loc[:, 'diff_occupation_cnt_fam_mebers_mean'] = data[feat_name] - data['CNT_FAM_MEMBERS']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')         
        data.loc[:, 'diff_occupation_days_birth_mean'] = data[feat_name] - data['DAYS_BIRTH']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')         
        data.loc[:, 'diff_occupation_days_employed_mean'] = data[feat_name] - data['DAYS_EMPLOYED']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')         
        data.loc[:, 'diff_occupation_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')         
        data.loc[:, 'diff_occupation_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')         
        data.loc[:, 'diff_occupation_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']
        
        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'OWN_CAR_AGE', np.mean, 'mean')         
        data.loc[:, 'diff_occupation_own_car_age_mean'] = data[feat_name] - data['OWN_CAR_AGE']

        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'YEARS_BUILD_AVG', np.mean, 'mean')         
        data.loc[:, 'diff_occupation_year_build_mean'] = data[feat_name] - data['YEARS_BUILD_AVG']

        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'ratio_annuity_credit', np.mean, 'mean')
        data.loc[:, 'diff_occupation_annuity_credit_mean'] = data[feat_name] - data['ratio_annuity_credit']

        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_occupation_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_occupation_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']

        data, feat_name = self.feature_interaction(data, ['OCCUPATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_occupation_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']


        # Organization type and other features
        data, feat_name = self.feature_interaction(data, ['ORGANIZATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_organization_ext_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']
        
        data, feat_name = self.feature_interaction(data, ['ORGANIZATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_organization_ext_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['ORGANIZATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_organization_ext_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['ORGANIZATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_organization_age_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['ORGANIZATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_organization_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['ORGANIZATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_organization_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['ORGANIZATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_organization_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']

        data, feat_name = self.feature_interaction(data, ['ORGANIZATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_organization_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']


        # INCOME Type and other features
        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_income_ext_source_1_mean']  = data[feat_name] - data['EXT_SOURCE_1']
        data.loc[:, 'ratio_income_ext_source_1_mean'] = data[feat_name] / data['EXT_SOURCE_1'] 
        
        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_income_ext_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_income_ext_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_income_ext_age_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_income_ext_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_income_ext_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_income_ext_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']

        data, feat_name = self.feature_interaction(data, ['NAME_INCOME_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_income_ext_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']


        # EDUCATION Type and other features
        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_education_ext_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']
        
        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_education_ext_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_education_ext_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_education_ext_age_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_education_ext_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_education_ext_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_education_ext_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']

        data, feat_name = self.feature_interaction(data, ['NAME_EDUCATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_education_ext_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']

        # Family Type and Income Type
        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_family_income_ext_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_family_income_ext_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_family_income_ext_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_family_income_ext_age_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_family_income_ext_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_family_income_ext_credit_mean'] = data[feat_name] - data['AMT_CREDIT']
        
        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_family_income_ext_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']
        
        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_family_income_ext_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']
        
        
        # Family Type and Education Type
        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_family_education_ext_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_family_education_ext_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_family_education_ext_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_family_education_ext_age_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_family_education_ext_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_family_education_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_family_education_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_family_education_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']

        # Family Type, Organization Type
        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'ORGANIZATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_family_organization_ext_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'ORGANIZATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_family_organization_ext_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'ORGANIZATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_family_organization_ext_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'ORGANIZATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_family_organization_ext_age_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'ORGANIZATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_family_organization_ext_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'ORGANIZATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_family_organization_ext_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'ORGANIZATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_family_organization_ext_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'ORGANIZATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_family_organization_ext_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']


        # Family Type, Occupation Type
        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'OCCUPATION_TYPE'], 'EXT_SOURCE_1', np.mean, 'mean')
        data.loc[:, 'diff_family_occupation_ext_source_1_mean'] = data[feat_name] - data['EXT_SOURCE_1']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'OCCUPATION_TYPE'], 'EXT_SOURCE_2', np.mean, 'mean')
        data.loc[:, 'diff_family_occupation_ext_source_2_mean'] = data[feat_name] - data['EXT_SOURCE_2']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'OCCUPATION_TYPE'], 'EXT_SOURCE_3', np.mean, 'mean')
        data.loc[:, 'diff_family_occupation_ext_source_3_mean'] = data[feat_name] - data['EXT_SOURCE_3']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'OCCUPATION_TYPE'], 'DAYS_BIRTH', np.mean, 'mean')
        data.loc[:, 'diff_family_occupation_ext_age_mean'] = data[feat_name] - data['DAYS_BIRTH']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'OCCUPATION_TYPE'], 'DAYS_EMPLOYED', np.mean, 'mean')
        data.loc[:, 'diff_family_occupation_ext_empl_mean'] = data[feat_name] - data['DAYS_EMPLOYED']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'OCCUPATION_TYPE'], 'AMT_CREDIT', np.mean, 'mean')
        data.loc[:, 'diff_family_occupation_ext_credit_mean'] = data[feat_name] - data['AMT_CREDIT']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'OCCUPATION_TYPE'], 'AMT_ANNUITY', np.mean, 'mean')
        data.loc[:, 'diff_family_occupation_ext_annuity_mean'] = data[feat_name] - data['AMT_ANNUITY']

        data, feat_name = self.feature_interaction(data, ['NAME_FAMILY_STATUS', 'OCCUPATION_TYPE'], 'AMT_INCOME_TOTAL', np.mean, 'mean')
        data.loc[:, 'diff_family_occupation_ext_income_mean'] = data[feat_name] - data['AMT_INCOME_TOTAL']


        # frequency encoding of some of the categorical variables.
        data = frequency_encoding(data, FREQ_ENCODING_COLS)

        # add pca components
        if os.path.exists(os.path.join(basepath, self.params['output_path'] + f'{self.params["data_folder"]}pca.pkl')):
            pca_components = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + f'{self.params["data_folder"]}pca.pkl'))
        else:
            pca_components = super(Modelv132, self).add_pca_components(data.copy(), PCA_PARAMS)
            pca_components.to_pickle(os.path.join(basepath, self.params['output_path'] + f'{self.params["data_folder"]}pca.pkl'))
        
        # add tsne components
        if os.path.exists(os.path.join(basepath, self.params['output_path'] + f'{self.params["data_folder"]}tsne.pkl')):
            tsne_components = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + f'{self.params["data_folder"]}tsne.pkl'))
        else:
            tsne_components = super(Modelv132, self).add_tsne_components(data.copy())
            tsne_components.to_pickle(os.path.join(basepath, self.params['output_path'] + f'{self.params["data_folder"]}tsne.pkl'))
        

        pca_components.index = data.index
        data = pd.concat((data, pca_components), axis=1)

        tsne_components.index = data.index
        data = pd.concat((data, tsne_components), axis=1)


        # one hot encoding of some of the categorical variables controlled by a flag
        # if flag is True then one hot encoding else do frequency encoding.
        if compute_categorical == 'ohe':
            print('Computing One Hot Encoding of categorical features ...')
            print('*' * 100)

            data = super(Modelv132, self).prepare_ohe(data, OHE_COLS, drop_col=True)
        elif compute_categorical == 'freq':
            print('Computing Frequency Encoding of Categorical features ....')
            print('*' * 100)

            data = frequency_encoding(data, OHE_COLS)
        
        return data

    # This method would perform feature engineering on merged datasets.
    def fe(self, train, test, compute_categorical=None):
        original_train = train.copy()
        data           = self.get_features(original_train, test, compute_categorical)

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
        
        return super(Modelv132, self).train_lgb(X, y, Xte, yte, **params)

    # This method just takes in a model and test dataset and returns predictions 
    # prints out AUC on the test dataset as well in the process.
    def evaluate(self, test, feature_list, is_eval, model, TARGET_NAME='TARGET'):
        Xte = test.loc[:, feature_list]
        yte = []

        if is_eval:
            yte = test.loc[:, TARGET_NAME]

        return super(Modelv132, self).evaluate_lgb(Xte, yte, model)

    def cv_predict(self, train, test, feature_list, params, cv_adversarial_filepath=None, categorical_feature='auto'):
        return super(Modelv132, self).cv_predict(train, 
                                                    test,
                                                    feature_list, 
                                                    params, 
                                                    cv_adversarial_filepath=cv_adversarial_filepath,
                                                    categorical_feature=categorical_feature
                                                    )
    
    def predict_test(self, train, test, feature_list, params, n_folds=5):
        return super(Modelv132, self).predict_test_etc(train, test, feature_list, params, n_folds=n_folds)


    def cross_validate(self, train, feature_list, params, cv_adversarial_filepath=None, TARGET_NAME='TARGET'):
        Xtr = train.loc[:, feature_list]
        ytr = train.loc[:, TARGET_NAME]

        return super(Modelv132, self).cross_validate_etc(Xtr, ytr, params, cv_adversarial_filepath=cv_adversarial_filepath)


    def rf_fi(self, train, feature_list, SEED, target='TARGET'):
        X = train.loc[:, feature_list]
        y = train.loc[:, target]

        return super(Modelv132, self).rf_fi(X, y, SEED)
    
    def optimize_lgb(self, train, test, feature_list, TARGET_NAME='TARGET'):
        Xtr = train.loc[:, feature_list]
        ytr = train.loc[:, TARGET_NAME]

        Xte = test.loc[:, feature_list]
        yte = test.loc[:, TARGET_NAME]

        param_grid = {
            'sub_feature': (.01, .3),
            'max_depth': (3, 8),
            'min_data_in_leaf': (20, 100),
            'min_child_weight': (1, 100),
            'reg_lambda': (.1, 100),
            'reg_alpha': (.1, 100),
            'min_split_gain': (.01, .03),
            'num_leaves': (5, 100)
        }

        return super(Modelv132, self).optimize_lgb(Xtr, ytr, Xte, yte, param_grid)


    def get_oof_preds(self, train, test, feature_list, model, TARGET_NAME='TARGET'):
        X = train.loc[:, feature_list]
        y = train.loc[:, TARGET_NAME]
        
        Xte = test.loc[:, feature_list]

        return super(Modelv132, self).oof_preds(X, y, Xte, model)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Home Credit Default Risk Solution')
    
    parser.add_argument('-input_path', help='Path to input directory')     # path to raw files
    parser.add_argument('-output_path', help='Path to output directory')   # path to working data folder 
    parser.add_argument('-data_folder', help='Folder name of the dataset') # dataset folder name
    parser.add_argument('-p', type=bool, help='Preprocess')
    parser.add_argument('-cv', type=bool, help='Cross Validation')
    parser.add_argument('-cv_predict',type=bool, help='Cross Validation and Predictions for test set.')
    parser.add_argument('-v', type=str, help='Validation')
    parser.add_argument('-features', type=bool, help='Generate Features')
    parser.add_argument('-rf_fi', type=bool, help='Random Forest Classifier Feature Importance')
    parser.add_argument('-s', type=bool, help='Whether to work on a sample or not.')
    parser.add_argument('-seed', type=int, help='Random SEED')
    parser.add_argument('-cv_seed', type=int, help='CV SEED')
    parser.add_argument('-oof', type=bool, help='OOF preds for training and test set.')
    parser.add_argument('-t', type=bool, help='Full Training Loop.')
    parser.add_argument('-bo', type=bool, help='Hyper-Parameter Tuning using Bayesian Optimization')
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

        m  = Modelv132(**params)
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

        m = Modelv132(**params)
        m.prepare_features()

    elif args.v is not None and len(args.v):
        print('Train and generate predictions on a fold')

        input_path      = args.input_path
        output_path     = args.output_path
        data_folder     = args.data_folder
        fold_indicator  = args.v
        is_sample       = args.s
        cv_seed         = args.cv_seed
        SEED            = int(args.seed)

        print('*' * 100)
        print('SEED FOUND: {}'.format(SEED))

        params = {
            'input_path': input_path,
            'output_path': output_path,
            'data_folder': data_folder
        }

        PARAMS  = joblib.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{cv_seed}_params.pkl'))
        
        # Set seed to Params
        PARAMS['seed'] = SEED
        PARAMS['feature_fraction_seed'] = SEED
        PARAMS['bagging_seed'] = SEED
        PARAMS['early_stopping_rounds'] = None # explicitly make it None

        print('*' * 100)
        print('PARAMS: {}'.format(PARAMS))

        m   = Modelv132(**params)
            
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

        # ite    = pd.read_csv(os.path.join(basepath, input_path + 'cv_adversarial_idx_v1.csv'), usecols=[fold_indicator])[fold_indicator].values
        ite  = pd.read_csv(os.path.join(basepath, input_path + 'cv_idx_test_stratified.csv'), usecols=[fold_indicator])[fold_indicator].values
        print('Shape of fold indices ', len(ite))

        itr    = np.array(list(set(data.iloc[:m.n_train].index) - set(ite)))
        
        train    = data.loc[data.index.isin(itr)]
        test     = data.loc[data.index.isin(ite)]

        del data
        gc.collect()

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
                
                if not os.path.exists(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}{fold_indicator}_true_holdout.npy')):
                    np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}{fold_indicator}_true_holdout.npy'), test.TARGET)
                
                hold_preds = model.predict(test.loc[:, feature_list])
                np.save(os.path.join(basepath, output_path + f'{model_identifier}_preds_holdout.npy'), hold_preds)
                feat_df.to_csv(os.path.join(basepath, output_path + f'{model_identifier}_feat_imp.csv'), index=False)
    
    elif args.cv:
        print('Cross validation on training and store parameters and cv score on disk ...')

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

        m   = Modelv132(**params)
            
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

        del data
        gc.collect()
        
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

        cv_adversarial_filepath = os.path.join(basepath, 'data/raw/cv_idx_test_stratified.csv')

        mean_auc, std_auc = m.cross_validate(train, feature_list, PARAMS.copy(), cv_adversarial_filepath)
        cv_score   = str(mean_auc) + '_' + str(std_auc)
        
        
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

        m   = Modelv132(**params)
        

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

        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_test_preds.npy')):
            oof_train_preds  = np.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_oof_train_preds.npy'))       
            test_preds       = np.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_test_preds.npy'))
            test_preds_final = np.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_test_preds_final.npy'))

            auc             = joblib.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_oof_auc.pkl'))
        else:
            auc, oof_train_preds, test_preds, test_preds_final = m.predict_test(train, test, feature_list, PARAMS.copy())

            np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_oof_train_preds.npy'), oof_train_preds)
            np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_test_preds.npy'), test_preds)
            np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_test_preds_final.npy'), test_preds_final)

            joblib.dump(auc, os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_oof_auc.pkl'))
        
        sub_identifier = "%s-%s-%s-%s-%s" % (datetime.now().strftime('%Y%m%d-%H%M'), MODEL_FILENAME, auc, SEED, data_folder[:-1])

        # generate for test set
        sub            = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv.zip'))
        sub['TARGET']  = test_preds_final
        sub.to_csv(os.path.join(basepath, 'submissions/%s.csv'%(sub_identifier)), index=False)

    elif args.oof:
        print('Generate oof predictions for train and test set ...')
        
        input_path      = args.input_path
        output_path     = args.output_path
        data_folder     = args.data_folder
        SEED            = args.seed

        params = {
            'input_path': input_path,
            'output_path': output_path,
            'data_folder': data_folder
        }

        m   = Modelv132(**params)
            
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}data.h5')):
            print('Loading dataset from disk ...')
            data = pd.read_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')
        else:
            print('Merge feature groups and save them to disk ...')
            train, test  = m.merge_datasets()
            train, test  = m.fe(train, test)
            
            data         = pd.concat((train, test))
            data         = m.reduce_mem_usage(data)

            data.to_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')

            del train, test
            gc.collect()

        train  = data.iloc[:m.n_train]
        test   = data.iloc[m.n_train:]
        

        del data
        gc.collect()
        

        # check to see if feature list exists on disk or not for a particular model
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy')):
            feature_list = np.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy'))
        else: 
            feature_list = train.columns.tolist()
            feature_list = list(set(feature_list) - set(COLS_TO_REMOVE))
            np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy'), feature_list)

        PARAMS = joblib.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{SEED}_params.pkl'))    

        # model construction
        model = lgb.LGBMClassifier(num_leaves=PARAMS['num_leaves'],
                                   max_depth=PARAMS['max_depth'],
                                   learning_rate=PARAMS['learning_rate'],
                                   n_estimators=PARAMS['num_boost_round'],
                                   objective=PARAMS['objective'],
                                   min_child_weight=PARAMS['min_child_weight'],
                                   min_child_samples=PARAMS['min_data_in_leaf'],
                                   subsample=PARAMS['bagging_fraction'],
                                   colsample_bytree=PARAMS['sub_feature'],
                                   reg_lambda=PARAMS['reg_lambda'],
                                   random_state=SEED,
                                   verbose=-1,
                                   n_jobs=8
                                   )
        
        oof_preds, test_preds = m.get_oof_preds(train, test, feature_list, model)

        np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{SEED}_oof_preds.npy'), oof_preds)
        np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{SEED}_test.npy'), test_preds)

        
    elif args.t:
        print('Full Training')

        input_path      = args.input_path
        output_path     = args.output_path
        data_folder     = args.data_folder
        CV_SEED         = args.cv_seed
        SEED            = args.seed
        
        params = {
            'input_path': input_path,
            'output_path': output_path,
            'data_folder': data_folder
        }

        m   = Modelv132(**params)
        
        # Load or save data from/ on disk
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}data.h5')):
            print('Loading dataset from disk ...')
            data = pd.read_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')
            
        else:
            print('Merge feature groups and save them to disk ...')
            train, test  = m.merge_datasets()
            train, test  = m.fe(train, test)
            
            data         = pd.concat((train, test))
            data         = m.reduce_mem_usage(data)

            data.to_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')
            
            del train, test
            gc.collect()

        # separate out training and test set.
        train  = data.iloc[:m.n_train]
        test   = data.iloc[m.n_train:]

        # check to see if feature list exists on disk or not for a particular model
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy')):
            feature_list = np.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy'))
        else: 
            feature_list = train.columns.tolist()
            feature_list = list(set(feature_list) - set(COLS_TO_REMOVE))
            np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy'), feature_list)

        
        # Load params and holdout score from disk.
        PARAMS        = joblib.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_params.pkl'))
        HOLDOUT_SCORE = joblib.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_{CV_SEED}_cv.pkl'))

        PARAMS['num_boost_round'] = int(1.1 * PARAMS['num_boost_round'])
        PARAMS['learning_rate']  /= 1.1

        PARAMS['seed'] = SEED
        PARAMS['feature_fraction_seed'] = SEED
        PARAMS['bagging_seed'] = SEED

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

    elif args.rf_fi:
        print('Train a random forest classifier and save feature importance on disk ...')

        input_path      = args.input_path
        output_path     = args.output_path
        data_folder     = args.data_folder
        SEED            = args.seed

        params = {
            'input_path': input_path,
            'output_path': output_path,
            'data_folder': data_folder
        }

        m   = Modelv132(**params)
        
        # Loading data
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}data.h5')):
            print('Loading dataset from disk ...')
            data = pd.read_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')
        else:
            print('Merge feature groups and save them to disk ...')
            train, test  = m.merge_datasets()
            train, test  = m.fe(train, test)
            
            data         = pd.concat((train, test))
            data         = m.reduce_mem_usage(data)

            data.to_hdf(os.path.join(basepath, output_path + f'{data_folder}data.h5'), format='table', key='data')

            del train, test
            gc.collect()

        train  = data.iloc[:m.n_train]
        
        del data
        gc.collect()
        
        # check to see if feature list exists on disk or not for a particular model
        if os.path.exists(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy')):
            feature_list = np.load(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy'))
        else: 
            feature_list = train.columns.tolist()
            feature_list = list(set(feature_list) - set(COLS_TO_REMOVE))
            np.save(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_features.npy'), feature_list)
        

        feat_df = m.rf_fi(train, feature_list, SEED)
        feat_df.to_csv(os.path.join(basepath, output_path + f'{data_folder}{MODEL_FILENAME}_rf_fi.csv'), index=False)