import pandas as pd
import numpy as np

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

COLS_TO_REMOVE = ['SK_ID_CURR', 'TARGET',
                  'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_5',
                  'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_11',
                  'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_17',
                  'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_4',
                  'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_21',
                  'FLAG_DOCUMENT_10',          
                  'AMT_REQ_CREDIT_BUREAU_DAY', 'FLAG_DOCUMENT_2',
                  'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_19',
                  'FLAG_CONT_MOBILE', 'FLAG_MOBIL',
                  'FLAG_DOCUMENT_12', 'REG_REGION_NOT_LIVE_REGION',
                  'FLAG_DOCUMENT_13', 'AMT_REQ_CREDIT_BUREAU_HOUR'
                ]

PARAMS = {
    'num_boost_round': 5000,
    'objective': 'binary',
    'learning_rate': .02,
    'metric': 'auc',
    'min_data_in_leaf': 100,
    'num_leaves': 60,
    'feature_fraction': .3,
    'feature_fraction_seed': SEED,
    'lambda_l1': 5,
    'lambda_l2': 20,
    'min_child_weight': 2.,
    'nthread': 4,
    'seed': SEED
}

MODEL_FILENAME = 'v34_model.txt'


class ModelV34(BaseModel):
    def __init__(self, **params):
        self.params  = params
        self.n_train = 307511
    
    def load_data(self, filenames):
        dfs = []
        
        for filename in filenames:
            dfs.append(pd.read_csv(filename, parse_dates=True, keep_date_col=True))
        
        df       = pd.concat(dfs)
        df.index = np.arange(len(df))
        df       = super(ModelV34, self).reduce_mem_usage(df)

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
        dtr, dte = super(ModelV34, self).create_fold(data, seed)

        dtr.index = np.arange(len(dtr))
        dte.index = np.arange(len(dte))

        dtr.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'application_{fold_name}train.pkl'))
        dte.to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'application_{fold_name}test.pkl'))

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

            del installments, prev_app
            gc.collect()

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_installments_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'prev_app_installments_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

        else:
            print('Already generated features based on Previous application and Installment Payments.')
        
        if not os.path.exists(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'loan_stacking_{fold_name}train.pkl')):
            bureau = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + 'bureau.pkl'))

            for col in bureau.select_dtypes(include=['category']).columns:
                bureau.loc[:, col] = bureau.loc[:, col].cat.codes

            print('Generating features based on loan stacking ....')

            t0                  = time.clock()
            data, FEATURE_NAMES = loan_stacking(bureau, data)
            data.index          = np.arange(len(data))

            # fill infrequent values
            data = super(ModelV34, self).fill_infrequent_values(data)

            data.iloc[:ntrain].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'loan_stacking_{fold_name}train.pkl'))
            data.iloc[ntrain:].loc[:, FEATURE_NAMES].to_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'loan_stacking_{fold_name}test.pkl'))
            print('\nTook: {} seconds'.format(time.clock() - t0))

            del bureau
            gc.collect()
        else:
            print('Already generated features based on loan stacking.')


    def merge_datasets(self, fold_name):
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
                     f'loan_stacking_{fold_name}'
                    ]

        train  = []
        test   = []

        for filename_ in filenames:
            tmp       = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'{filename_}train.pkl'))
            tmp.index = np.arange(len(tmp)) 
            train.append(tmp)

        for filename_ in filenames:
            tmp       = pd.read_pickle(os.path.join(basepath, self.params['output_path'] + self.params['run_name'] + f'{filename_}test.pkl'))
            tmp.index = np.arange(len(tmp))
            test.append(tmp)

        return pd.concat(train, axis=1), pd.concat(test, axis=1)

    def train(self, train, test, feature_list, is_eval, **params):
        X = train.loc[:, feature_list]
        y = train.loc[:, 'TARGET']
        
        Xte = test.loc[:, feature_list]
        yte = []

        if is_eval:
            yte = test.loc[:, 'TARGET']
        
        return super(ModelV34, self).train_lgb(X, y, Xte, yte, **params)

    def evaluate(self, test, feature_list, is_eval, model):
        Xte = test.loc[:, feature_list]
        yte = []

        if is_eval:
            yte = test.loc[:, 'TARGET']

        return super(ModelV34, self).evaluate_lgb(Xte, yte, model)



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
    parser.add_argument('-seed', type=int, help='Random Seed')
    parser.add_argument('-fold_name', help='Fold Name')
    

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

        m  = ModelV34(**params)
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

        m = ModelV34(**params)
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

        m = ModelV34(**params)
        m.prepare_features(fold_name)

    elif args.v:
        print('Training model and tuning parameters using a single fold ....')
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

        m              = ModelV34(**params)
        train, test    = m.merge_datasets(fold_name)
        feature_list   = list(set(train.columns) - set(COLS_TO_REMOVE))
        is_eval        = len(fold_name) > 0

        if not is_eval:
            PARAMS['num_boost_round'] = int(1.2 * PARAMS['num_boost_round'])
            PARAMS['learning_rate']   /= 1.2

        m.train(train, test, feature_list, is_eval, MODEL_FILENAME, **PARAMS)
    

    elif args.t:
        print('Train Model and evaluate predictions on a given fold ....')
        print()

        run_name        = args.run_name
        input_path      = args.input_path
        output_path     = args.output_path
        fold_name       = args.fold_name
        
        params = {
            'input_path': input_path,
            'output_path': output_path,
            'run_name': run_name
        }

        m              = ModelV34(**params)
        train, test    = m.merge_datasets(fold_name)

        if os.path.exists(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl')):
            feature_list = joblib.load(os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl'))
        else: 
            feature_list   = list(set(train.columns) - set(COLS_TO_REMOVE)) 
            joblib.dump(feature_list, os.path.join(basepath, output_path + run_name + f'{MODEL_FILENAME}_features.pkl'))

        # check to see if we are doing validation or final test generation.
        is_eval  = len(fold_name) > 0
        
        if not is_eval:
            # use best iteration found through different folds
            PARAMS['num_boost_round'] = 1200
            PARAMS['num_boost_round'] = int(1.2 * PARAMS['num_boost_round'])
            PARAMS['learning_rate']   /= 1.2

        t0 = time.clock()

        # train model
        model = m.train(train, test, feature_list, is_eval, **PARAMS)
        
        print('Took: {} seconds to train model'.format(time.clock() - t0))

        # evaluation part
        preds, score  = m.evaluate(test, feature_list, is_eval, model)
        
        # save submission
        if not is_eval:
            print('Generating Submissions ...')

            # found through validation scores across multiple folds
            HOLDOUT_SCORE = (0.7913 + .7926) / 2

            sub_identifier = "%s-%s-%.5f" % (datetime.now().strftime('%Y%m%d-%H%M'), MODEL_FILENAME, HOLDOUT_SCORE)

            sub            = pd.read_csv(os.path.join(basepath, 'data/raw/sample_submission.csv.zip'))
            sub['TARGET']  = preds

            sub.to_csv(os.path.join(basepath, 'submissions/%s.csv'%(sub_identifier)), index=False)
        else:
            # save oof predictions
            joblib.dump(preds, os.path.join(basepath, output_path + run_name + f'preds_{MODEL_FILENAME}_{fold_name}_{score}.pkl'))
