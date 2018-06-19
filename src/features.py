import pandas as pd
import numpy as np

import gc

def frequency_encoding(data, cols):
    for col in cols:
        data.loc[:, col] = data.groupby(col)[col].transform(lambda x: len(x)).values
    return data

def get_agg_features(data, gp, f, on):
    agg         = gp.groupby(on)[f]\
                        .agg({np.mean, 
                              np.median, 
                              np.max, 
                              np.min, 
                              np.var, 
                              np.sum}).fillna(-1)
    
    for c in agg.select_dtypes(include=['float64']).columns:
        agg[c]  = agg[c].astype(np.float32)
        
    cols        = [f'{f}_{c}' for i, c in enumerate(agg.columns)]
    agg.columns = cols
    agg         = agg.reset_index()
    data        = data.merge(agg, on=on, how='left')
    
    del agg
    gc.collect()
    
    return data, cols    

def log_features(data, features):
    for f in features:
        data.loc[:, f] = data[f].map(lambda x: np.log(x + 1)).replace([np.inf, -np.inf], np.nan)
    
    return data


def current_application_features(data):
    # feature names
    FEATURE_NAMES = []

    # deviation in three external scores
    data.loc[:, 'EXT_SOURCE_DEV']  = data.loc[:, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].apply(np.std, axis=1).astype(np.float32)
    FEATURE_NAMES += ['EXT_SOURCE_DEV']

    # sum of external scores
    data.loc[:, 'EXT_SOURCE_SUM'] = data['EXT_SOURCE_1'].fillna(0) + data['EXT_SOURCE_2'].fillna(0) + data['EXT_SOURCE_3'].fillna(0)
    FEATURE_NAMES += ['EXT_SOURCE_SUM']
    
    # mean of external scores
    data.loc[:, 'MEAN_EXTERNAL_SCORE'] = (data['EXT_SOURCE_1'].fillna(0) + data['EXT_SOURCE_2'].fillna(0) + data['EXT_SOURCE_3'].fillna(0)) / 3
    FEATURE_NAMES += ['MEAN_EXTERNAL_SCORE']
    
    # feature interactions
    data.loc[:, 'EXT_3_2'] = data.loc[:, 'EXT_SOURCE_3'] * data.loc[:, 'EXT_SOURCE_2']
    data.loc[:, 'EXT_1_3'] = data.loc[:, 'EXT_SOURCE_1'] * data.loc[:, 'EXT_SOURCE_3']
    data.loc[:, 'EXT_1_2'] = data.loc[:, 'EXT_SOURCE_1'] * data.loc[:, 'EXT_SOURCE_2']

    data.loc[:, 'EXT_1_2_sum'] = data.loc[:, 'EXT_SOURCE_1'] + data.loc[:, 'EXT_SOURCE_2']
    data.loc[:, 'EXT_1_3_sum'] = data.loc[:, 'EXT_SOURCE_1'] + data.loc[:, 'EXT_SOURCE_3']
    data.loc[:, 'EXT_2_3_sum'] = data.loc[:, 'EXT_SOURCE_2'] + data.loc[:, 'EXT_SOURCE_3']
    
    data.loc[:, 'EXT_1_2_div'] = data.loc[:, 'EXT_SOURCE_1'] / data.loc[:, 'EXT_SOURCE_2']
    data.loc[:, 'EXT_1_3_div'] = data.loc[:, 'EXT_SOURCE_1'] / data.loc[:, 'EXT_SOURCE_3']
    data.loc[:, 'EXT_2_3_div'] = data.loc[:, 'EXT_SOURCE_2'] / data.loc[:, 'EXT_SOURCE_3']
    

    FEATURE_NAMES += ['EXT_3_2', 
                      'EXT_1_3', 
                      'EXT_1_2',
                      'EXT_1_2_sum', 
                      'EXT_1_3_sum', 
                      'EXT_2_3_sum',
                      'EXT_1_2_div', 
                      'EXT_1_3_div', 
                      'EXT_2_3_div'
                     ]

    # treat 365243 in days employed as null value
    data.loc[:, 'DAYS_EMPLOYED'] = data.DAYS_EMPLOYED.replace({365243: np.nan})

    # number of null values in external scores
    data.loc[:, 'NUM_NULLS_EXT_SCORES'] = data.EXT_SOURCE_1.isnull().astype(np.int8) +\
                                          data.EXT_SOURCE_2.isnull().astype(np.int8) +\
                                          data.EXT_SOURCE_3.isnull().astype(np.int8)
    FEATURE_NAMES += ['NUM_NULLS_EXT_SCORES']

    # relationship between amount credit and total income
    data.loc[:, 'ratio_credit_income'] = data.loc[:, 'AMT_CREDIT'] / data.loc[:, 'AMT_INCOME_TOTAL']

    # relationship between annual amount to be paid and income
    data.loc[:, 'ratio_annuity_income'] = data.loc[:, 'AMT_ANNUITY'] / data.loc[:, 'AMT_INCOME_TOTAL']

    # relationship between amount annuity and age
    data.loc[:, 'ratio_annuity_age'] = (data.loc[:, 'AMT_ANNUITY'] / (-data.loc[:, 'DAYS_BIRTH'] / 365)).astype(np.float32)
    FEATURE_NAMES += ['ratio_credit_income', 'ratio_annuity_income', 'ratio_annuity_age']
    
    # number of missing values in an application
    data.loc[:, 'num_missing_values'] = data.loc[:, data.columns.drop('TARGET')].isnull().sum(axis=1).values
    FEATURE_NAMES += ['num_missing_values']
    
    # feature interaction between age and days employed
    data.loc[:, 'age_plus_employed']  = data.loc[:, 'DAYS_BIRTH'] + data.loc[:, 'DAYS_EMPLOYED']
    data.loc[:, 'ratio_age_employed'] = ((data.DAYS_EMPLOYED) / (data.DAYS_BIRTH)).astype(np.float32)
    FEATURE_NAMES += ['age_plus_employed', 'ratio_age_employed']

    # convert age, days employed into categorical variable and then concatenate those two.
    age_categorical = pd.cut(-data.DAYS_BIRTH / 365, bins=20)
    emp_categorical = pd.cut((-data.DAYS_EMPLOYED / 365), bins=15)

    data.loc[:, 'age_emp_categorical'] = pd.factorize(age_categorical.astype(np.str) + '_' + emp_categorical.astype(np.str))[0]
    FEATURE_NAMES += ['age_emp_categorical']
    
    # interaction between occupation type and age
    data.loc[:, 'age_occupation'] = pd.factorize(age_categorical.astype(np.str) + '_' + data.OCCUPATION_TYPE.astype(np.str))[0]
    FEATURE_NAMES += ['age_occupation']

    # interaction between occupation type and employment
    data.loc[:, 'emp_occupation'] = pd.factorize(emp_categorical.astype(np.str) + '_' + data.OCCUPATION_TYPE.astype(np.str))[0]
    FEATURE_NAMES += ['emp_occupation']
    
    del age_categorical, emp_categorical
    gc.collect()

    # ratio of value of goods against which loan is given to total income
    data.loc[:, 'ratio_goods_income'] = data.loc[:, 'AMT_GOODS_PRICE'] / data.loc[:, 'AMT_INCOME_TOTAL']
    FEATURE_NAMES += ['ratio_goods_income']
    
    # feature interaction between value of goods against which loan is given to annual loan amount to be paid
    data.loc[:, 'ratio_goods_annuity'] = data.loc[:, 'AMT_GOODS_PRICE'] / data.loc[:, 'AMT_ANNUITY']
    data.loc[:, 'mult_goods_annuity']  = data.loc[:, 'AMT_GOODS_PRICE'] * data.loc[:, 'AMT_ANNUITY']
    FEATURE_NAMES += ['ratio_goods_annuity', 'mult_goods_annuity']
    
    # feature interaction value of goods and amount credit
    data.loc[:, 'ratio_goods_credit'] = (data.loc[:, 'AMT_GOODS_PRICE'] / data.loc[:, 'AMT_CREDIT']).replace([np.inf, -np.inf], np.nan)
    data.loc[:, 'mult_goods_credit']  = (data.loc[:, 'AMT_GOODS_PRICE'] * data.loc[:, 'AMT_CREDIT']).replace([np.inf, -np.inf], np.nan)
    FEATURE_NAMES += ['ratio_goods_credit', 'mult_goods_credit']

    # feature interaction between annuity and amount credit
    data.loc[:, 'ratio_annuity_credit'] = data.loc[:, 'AMT_CREDIT'] / data.loc[:, 'AMT_ANNUITY'].replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # feature interaction between amount credit and age
    data.loc[:, 'ratio_credit_age'] = (data.AMT_CREDIT / (-data.DAYS_BIRTH / 365)).astype(np.float32)

    # feature interaction between amount credit and days before application id was changed
    data.loc[:, 'ratio_credit_id_change'] = (data.AMT_CREDIT / -data.DAYS_ID_PUBLISH).replace([np.inf, -np.inf], np.nan)

    # feature interaction between days id publish and age
    data.loc[:, 'ratio_id_change_age'] = ((data.DAYS_ID_PUBLISH / (-data.DAYS_BIRTH / 365))).astype(np.float32)
    FEATURE_NAMES += ['ratio_annuity_credit', 'ratio_credit_age', 'ratio_credit_id_change', 'ratio_id_change_age']
    

    # ratio of annuity and external score
    data.loc[:, 'ratio_annuity_score_1'] = (data.loc[:, 'AMT_ANNUITY'] / data.loc[:, 'EXT_SOURCE_1']).replace([np.inf, -np.inf], np.nan)
    data.loc[:, 'ratio_annuity_score_2'] = (data.loc[:, 'AMT_ANNUITY'] / data.loc[:, 'EXT_SOURCE_2']).replace([np.inf, -np.inf], np.nan)
    data.loc[:, 'ratio_annuity_score_3'] = (data.loc[:, 'AMT_ANNUITY'] / data.loc[:, 'EXT_SOURCE_3']).replace([np.inf, -np.inf], np.nan)
    FEATURE_NAMES += ['ratio_annuity_score_1', 'ratio_annuity_score_2', 'ratio_annuity_score_3']
    

    # ratio of annuity, credit multiplied by external scores
    data.loc[:, 'ratio_credit_annuity_score_1'] = ((data.loc[:, 'AMT_ANNUITY'] / data.loc[:, 'AMT_CREDIT']) * data.loc[:, 'EXT_SOURCE_1']).replace([np.inf, -np.inf], np.nan)
    data.loc[:, 'ratio_credit_annuity_score_2'] = ((data.loc[:, 'AMT_ANNUITY'] / data.loc[:, 'AMT_CREDIT']) * data.loc[:, 'EXT_SOURCE_2']).replace([np.inf, -np.inf], np.nan)
    data.loc[:, 'ratio_credit_annuity_score_3'] = ((data.loc[:, 'AMT_ANNUITY'] / data.loc[:, 'AMT_CREDIT']) * data.loc[:, 'EXT_SOURCE_3']).replace([np.inf, -np.inf], np.nan)
    FEATURE_NAMES += ['ratio_credit_annuity_score_1', 'ratio_credit_annuity_score_2', 'ratio_credit_annuity_score_3']
    
    # ratio of owner's car age with his age
    data.loc[:, 'ratio_car_person_age'] = (data.OWN_CAR_AGE / -data.DAYS_BIRTH)

    # difference credit and income
    data.loc[:, 'diff_credit_income'] = data.AMT_CREDIT - data.AMT_INCOME_TOTAL

    # difference income total and annuity
    data.loc[:, 'diff_income_annuity']  = data.AMT_ANNUITY - data.AMT_INCOME_TOTAL

    # difference credit and goods price
    data.loc[:, 'diff_credit_goods'] = data.AMT_CREDIT - data.AMT_GOODS_PRICE
    FEATURE_NAMES += ['ratio_car_person_age', 'diff_credit_income', 'diff_income_annuity', 'diff_credit_goods']
    

    # max, mean, std of feature groups related to days before any document was modified or changed
    data.loc[:, 'max_document_modified'] = data.loc[:, ['DAYS_REGISTRATION',
                                            'DAYS_ID_PUBLISH',
                                            'DAYS_LAST_PHONE_CHANGE'
                                            ]].apply(np.max, axis=1)
        
    data.loc[:, 'mean_document_modified'] = data.loc[:, ['DAYS_REGISTRATION',
                                        'DAYS_ID_PUBLISH',
                                        'DAYS_LAST_PHONE_CHANGE'
                                        ]].apply(np.mean, axis=1)

    data.loc[:, 'std_document_modified'] = data.loc[:, ['DAYS_REGISTRATION',
                                        'DAYS_ID_PUBLISH',
                                        'DAYS_LAST_PHONE_CHANGE'
                                        ]].apply(np.std, axis=1)
        
    # combine feature groups
    data.loc[:, 'amt_reqd_summary']      = data.loc[:, [f for f in data.columns if 'AMT_REQ_CREDIT' in f]].apply(np.sum, axis=1)
    FEATURE_NAMES += ['max_document_modified', 
                      'mean_document_modified', 
                      'std_document_modified'
                      ]
    
    return data, FEATURE_NAMES

def bureau_features(bureau, data):
    COLS = data.columns.tolist()

    # number of previous loans for a particular user
    prev_num_loans = bureau.groupby('SK_ID_CURR').size()

    # number of previous active credits
    num_active_credits = bureau.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].sum()

    # aggregation features
    data, dc_cols  = get_agg_features(data, bureau, 'DAYS_CREDIT', 'SK_ID_CURR')
    data, acm_cols = get_agg_features(data, bureau, 'AMT_CREDIT_SUM', 'SK_ID_CURR')

    # logarithm of features
    data  = log_features(data, acm_cols)

    # mean number of days of CB credit at the time of application
    mean_days_credit_end = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].mean()

    # mean of maximum amount overdue on any credit line
    mean_max_amt_overdue = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].mean().map(lambda x: np.log(x + 1)).astype(np.float32)

    # mean of total amount overdue on any credit line
    mean_total_amt_overdue = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].mean().map(lambda x: np.log(x + 1)).astype(np.float32)

    # sum of num times credit was prolonged
    sum_num_times_prolonged = bureau.groupby('SK_ID_CURR')['CNT_CREDIT_PROLONG'].sum()

    # number of different types of credit taken from CREDIT BUREAU
    num_diff_credits = bureau.groupby('SK_ID_CURR')['CREDIT_TYPE'].nunique().astype(np.float32)

    # mean number of days of last credit update
    mean_days_credit_update = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].mean().astype(np.float32)   
        
    # summary of amount of annuity of credit bureau loans
    mean_cb_credit_annuity = bureau.groupby('SK_ID_CURR')['AMT_ANNUITY'].mean().map(lambda x: np.log(x + 1)).astype(np.float32).astype(np.float32)
    std_cb_credit_annuity  = bureau.groupby('SK_ID_CURR')['AMT_ANNUITY'].std().map(lambda x: np.log(x + 1)).astype(np.float32).astype(np.float32)

    # latest application reported to Home Credit
    latest_credit = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].max()
    data.loc[:, 'latest_credit'] = data.SK_ID_CURR.map(latest_credit).astype(np.float32)

    # credit duration
    res = bureau.loc[(bureau.CREDIT_ACTIVE == 0) &\
           (bureau.DAYS_CREDIT > -400) &\
           (bureau.DAYS_CREDIT_ENDDATE > 0)
           , ['SK_ID_CURR', 'SK_ID_BUREAU', 
              'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 
              'AMT_CREDIT_SUM']]

    res.loc[:, 'duration'] = (res.DAYS_CREDIT_ENDDATE - res.DAYS_CREDIT).astype(np.int32)
    res = res.groupby('SK_ID_CURR')['duration'].min()
    data.loc[:, 'credit_duration'] = data.SK_ID_CURR.map(res)

    del res
    gc.collect()

    # duration of close credits taken from Home Credit
    res  = bureau.loc[(bureau.CREDIT_ACTIVE == 2) & (bureau.DAYS_CREDIT_ENDDATE < 0), ['SK_ID_CURR', 'DAYS_CREDIT', 'DAYS_ENDDATE_FACT']]
    d2   = -(res.DAYS_CREDIT - res.DAYS_ENDDATE_FACT).astype(np.float32)

    d2   = d2.groupby(res.SK_ID_CURR).mean().astype(np.float32)
    data.loc[:, 'closed_credit_duration']  = data.SK_ID_CURR.map(d2).astype(np.float32)

    del d2, res
    gc.collect()
    
    # ratio of active to closed loan duration
    data.loc[:, 'div_deltas'] = (data.credit_duration / data.closed_credit_duration).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # deviation in difference between remaining duration of credit and how long before we applied for this credit
    diff_prev_curr_credit = bureau.DAYS_CREDIT_ENDDATE.fillna(0) - bureau.DAYS_CREDIT.fillna(0)
    diff_prev_curr_credit = diff_prev_curr_credit.groupby(bureau.SK_ID_CURR).std()
    data.loc[:, 'std_diff_prev_curr_credit'] = data.SK_ID_CURR.map(diff_prev_curr_credit).astype(np.float32)


    # mean of difference between remaining duration of credit and how long before we applied for this credit
    diff_prev_curr_credit = (bureau.DAYS_CREDIT_ENDDATE - bureau.DAYS_CREDIT).astype(np.float32)
    diff_prev_curr_credit = diff_prev_curr_credit.groupby(bureau.SK_ID_CURR).mean()
    data.loc[:, 'mean_diff_prev_curr_credit'] = data.SK_ID_CURR.map(diff_prev_curr_credit).astype(np.float32)

    # difference between last ended loan and most recent applied loan
    res = bureau.loc[bureau.CREDIT_ACTIVE == 2, ['SK_ID_CURR', 'DAYS_ENDDATE_FACT']]
    res = res.groupby('SK_ID_CURR')['DAYS_ENDDATE_FACT'].max() # most recently ended credit for a client

    tmp = bureau.loc[bureau.CREDIT_ACTIVE == 0, ['SK_ID_CURR', 'DAYS_CREDIT']]
    tmp = tmp.groupby('SK_ID_CURR')['DAYS_CREDIT'].max() # most recently applied credit at Home credit

    res   = tmp.add(-res, fill_value=np.nan).astype(np.float32)
    
    data.loc[:, 'mean_diff_ended_curr_credit'] = data.SK_ID_CURR.map(res)

    del res, tmp
    gc.collect()

    # difference between last ended loan and term of longest loan
    res = bureau.loc[bureau.CREDIT_ACTIVE == 2, ['SK_ID_CURR', 'DAYS_ENDDATE_FACT']]
    res = res.groupby('SK_ID_CURR')['DAYS_ENDDATE_FACT'].max() # most recently ended credit for a client

    tmp   = bureau.loc[bureau.CREDIT_ACTIVE == 0, ['SK_ID_CURR', 'DAYS_CREDIT_ENDDATE']]
    tmp   = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].max() # active credit to be expired soon

    res  = tmp.add(-res, fill_value=np.nan).astype(np.float32)
    data.loc[:, 'mean_diff_prev_remaining_credit']  = data.SK_ID_CURR.map(res)

    del res, tmp
    gc.collect()

    # mean of ratio of two differences
    diff1 = bureau.DAYS_ENDDATE_FACT - bureau.DAYS_CREDIT
    diff2 = bureau.DAYS_CREDIT_ENDDATE - bureau.DAYS_CREDIT
    diff  = (diff1 / diff2).replace([np.inf, -np.inf], np.nan)
    diff  = diff.groupby(bureau.SK_ID_CURR).mean()
    data.loc[:, 'ratio_two_diff'] = data.SK_ID_CURR.map(diff).astype(np.float32)

    num_nulls_enddate                = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].apply(lambda x: x.isnull().sum())
    data.loc[:, 'num_nulls_enddate'] = data.SK_ID_CURR.map(num_nulls_enddate).fillna(-99).astype(np.int8)

    # ratio of debt to total credit sum
    res         = bureau.loc[(bureau.CREDIT_ACTIVE == 0) & (bureau.AMT_CREDIT_SUM_DEBT > 0), ['SK_ID_CURR', 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM']]
    total_sum   = res.groupby(res.SK_ID_CURR)['AMT_CREDIT_SUM'].sum().astype(np.float32)
    total_debt  = res.groupby(res.SK_ID_CURR)['AMT_CREDIT_SUM_DEBT'].sum().astype(np.float32)
    tmp         = total_debt.div(total_sum, fill_value=np.nan).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    data.loc[:, 'ratio_debt_total'] = data.SK_ID_CURR.map(tmp)
    
    del res, tmp, total_sum, total_debt
    gc.collect()

    # ratio of total debt to income
    res = bureau.loc[(bureau.CREDIT_ACTIVE == 0) & (bureau.AMT_CREDIT_SUM_DEBT > 0), ['SK_ID_CURR', 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM']]
    total_debt = res.groupby(res.SK_ID_CURR)['AMT_CREDIT_SUM_DEBT'].sum().astype(np.float32)

    res = data.SK_ID_CURR.map(total_debt)
    data.loc[:, 'total_debt_to_income'] = (res / data.AMT_INCOME_TOTAL).replace([np.inf, -np.inf], np.nan)

    del res, total_debt
    gc.collect()

    # merge back with original dataframe
    data.loc[:, 'num_prev_loans']           = data.SK_ID_CURR.map(prev_num_loans).fillna(0).astype(np.float32).values
    data.loc[:, 'num_prev_active_credits']  = data.SK_ID_CURR.map(num_active_credits).fillna(0).astype(np.float32).values
    data.loc[:, 'mean_days_credit_end']     = data.SK_ID_CURR.map(mean_days_credit_end).fillna(0).values
    data.loc[:, 'mean_max_amt_overdue']     = data.SK_ID_CURR.map(mean_max_amt_overdue).fillna(0).values
    data.loc[:, 'mean_total_amt_overdue']   = data.SK_ID_CURR.map(mean_total_amt_overdue).values
    data.loc[:, 'sum_num_times_prolonged']  = data.SK_ID_CURR.map(sum_num_times_prolonged).fillna(0).astype(np.int8).values
    data.loc[:, 'mean_cb_credit_annuity']   = data.SK_ID_CURR.map(mean_cb_credit_annuity).fillna(0).values
    data.loc[:, 'std_cb_credit_annuity']    = data.SK_ID_CURR.map(std_cb_credit_annuity).fillna(0).values
    data.loc[:, 'num_diff_credits']         = data.SK_ID_CURR.map(num_diff_credits).fillna(0).values
    data.loc[:, 'mean_days_credit_update']  = data.SK_ID_CURR.map(mean_days_credit_update).fillna(0).values

    del prev_num_loans, num_active_credits, mean_days_credit_end
    del mean_max_amt_overdue, mean_total_amt_overdue, sum_num_times_prolonged
    del mean_cb_credit_annuity, std_cb_credit_annuity, num_diff_credits
    del mean_days_credit_update

    gc.collect()

    # number of active loans reported to Home Credit for a person in last n number of days
    res = bureau.loc[(bureau.CREDIT_ACTIVE == 0) &\
           (bureau.DAYS_CREDIT > -400)
           , ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']]\
            .groupby('SK_ID_CURR').size()
    
    data.loc[:, 'recent_bureau_loans'] = data.SK_ID_CURR.map(res).fillna(-1).astype(np.int8)

    del res
    gc.collect()

    # comparison of oldest loan with employment status
    oldest_credit = bureau.loc[bureau.DAYS_ENDDATE_FACT.isnull(), :].groupby('SK_ID_CURR')['DAYS_CREDIT'].min()
    oldest_credit = data.SK_ID_CURR.map(oldest_credit)
    data.loc[:, 'oldest_loan_employment'] = oldest_credit - data.DAYS_EMPLOYED.replace({365243: np.nan})

    # comparison of oldest loan with age status
    oldest_credit = bureau.loc[bureau.DAYS_ENDDATE_FACT.isnull(), :].groupby('SK_ID_CURR')['DAYS_CREDIT'].min()
    oldest_credit = data.SK_ID_CURR.map(oldest_credit)
    data.loc[:, 'oldest_loan_age'] = oldest_credit - data.DAYS_BIRTH

    del oldest_credit
    gc.collect()


    return data, list(set(data.columns) - set(COLS))

def feature_interactions(data):
    COLS = data.columns.tolist()
    
    # feature interaction between credit bureau annuity, current annuity and total income
    data.loc[:, 'ratio_cb_goods_annuity'] = (data.AMT_GOODS_PRICE / (data.mean_cb_credit_annuity + data.AMT_ANNUITY)).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # feature interaction between mean days credit update and last time id was changed by user
    data.loc[:, 'ratio_update_id']        = (data.mean_days_credit_update / data.DAYS_ID_PUBLISH).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    # feature interaction between mean credit amount of previous credits with current credit
    data.loc[:, 'ratio_curr_prev_credit'] = (data.AMT_CREDIT / data.AMT_CREDIT_SUM_mean).astype(np.float32)

    # Relationship between age, loan annuity and credit amount of the loan.
    data.loc[:, 'age_ratio_credit_annuity'] = (-data.DAYS_BIRTH / 365) * (data.AMT_CREDIT / data.AMT_ANNUITY)

    # Relationship between external source 3, loan annuity and credit amount of the loan
    data.loc[:, 'ext_3_credit_annuity'] = (data.EXT_SOURCE_3) * (data.AMT_CREDIT / data.AMT_ANNUITY)

    return data, list(set(data.columns) - set(COLS))

def bureau_and_balance(bureau, bureau_bal, data):
    COLS = data.columns.tolist()

    res = bureau_bal.loc[bureau_bal.STATUS.isin([0, 1, 2, 3, 4, 5])].groupby('SK_ID_BUREAU').size().reset_index()
    res = res.merge(bureau.loc[:, ['SK_ID_CURR', 'SK_ID_BUREAU']]).drop('SK_ID_BUREAU', axis=1)
    res = res.groupby('SK_ID_CURR')[0].sum()
    data.loc[:, 'mean_status'] = data.SK_ID_CURR.map(res)

    del res
    gc.collect()

    return data, list(set(data.columns) - set(COLS))

def prev_app_features(prev_app, data):
    COLS = data.columns.tolist()
    
    # number of previous applications
    num_prev_apps                = prev_app.groupby('SK_ID_CURR').size()
    data.loc[:, 'num_prev_apps'] = data.SK_ID_CURR.map(num_prev_apps).fillna(0).astype(np.int8) 

    # mean amount to be paid annually for previous applications
    prev_app_mean_annuity        = prev_app.groupby('SK_ID_CURR')['AMT_ANNUITY'].mean().map(lambda x: np.log(x + 1))
    prev_app_mean_annuity        = data.SK_ID_CURR.map(prev_app_mean_annuity)

    # ratio of previous annuity to current annuity
    data.loc[:, 'ratio_prev_curr_annuity'] = (prev_app_mean_annuity / data.AMT_ANNUITY).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    data.loc[:, 'diff_prev_curr_annuity']  = (prev_app_mean_annuity - data.AMT_ANNUITY).replace([np.inf, -np.inf], np.nan).astype(np.float32)

    del num_prev_apps, prev_app_mean_annuity
    gc.collect()

    down_payment_to_application = prev_app.groupby('SK_ID_CURR').apply(lambda x: (x['AMT_DOWN_PAYMENT'].fillna(0) / x['AMT_APPLICATION']).sum())
    data.loc[:, 'down_payment_to_application'] = data.SK_ID_CURR.map(down_payment_to_application).astype(np.float32)

    del down_payment_to_application
    gc.collect()

    # mean interest rate on down payments of previous applications
    mean_down_payment_rate                = prev_app.groupby('SK_ID_CURR')['RATE_DOWN_PAYMENT'].mean()
    data.loc[:, 'mean_down_payment_rate'] = data.SK_ID_CURR.map(mean_down_payment_rate)

    del mean_down_payment_rate
    gc.collect()

    
    most_freq_rejection_reason = prev_app.groupby('SK_ID_CURR').apply(lambda x: x.CODE_REJECT_REASON.value_counts().index.values[0])
    data.loc[:, 'most_freq_rejection_reason'] = data.SK_ID_CURR.map(most_freq_rejection_reason).astype(np.float32)

    del most_freq_rejection_reason
    gc.collect()

    # median amount annuity
    median_annuity                = prev_app.groupby('SK_ID_CURR')['AMT_ANNUITY'].median().map(lambda x: np.log(x + 1))
    data.loc[:, 'median_annuity'] = data.SK_ID_CURR.map(median_annuity).astype(np.float32)

    del median_annuity
    gc.collect()

    # mean of past annuity to credit applications
    past_annuity_credit = (prev_app.AMT_ANNUITY / prev_app.AMT_CREDIT).replace([np.inf, -np.inf], np.nan)
    past_annuity_credit = past_annuity_credit.groupby(prev_app.SK_ID_CURR).mean()
    data.loc[:, 'past_annuity_to_credit_mean'] = data.SK_ID_CURR.map(past_annuity_credit).astype(np.float32)

    del past_annuity_credit
    gc.collect()

    # std of past annuity to credit applications

    res = (prev_app.AMT_ANNUITY / prev_app.AMT_CREDIT).replace([np.inf, -np.inf], np.nan)
    res = res.groupby(prev_app.SK_ID_CURR).std()
    data.loc[:, 'past_annuity_to_credit_std'] = data.SK_ID_CURR.map(res).astype(np.float32)

    # sum of past annuity to credit applications
    res = (prev_app.AMT_ANNUITY / prev_app.AMT_CREDIT).replace([np.inf, -np.inf], np.nan)
    res = res.groupby(prev_app.SK_ID_CURR).sum()
    data.loc[:, 'past_annuity_to_credit_sum'] = data.SK_ID_CURR.map(res).astype(np.float32)


    # difference of down_payment * rate and annuity
    diff_dp_annuity = ((prev_app.AMT_DOWN_PAYMENT * prev_app.RATE_DOWN_PAYMENT) - prev_app.AMT_ANNUITY).replace([np.inf, -np.inf])
    diff_dp_annuity = diff_dp_annuity.groupby(prev_app.SK_ID_CURR).sum()
    data.loc[:, 'diff_dp_annuity'] = data.SK_ID_CURR.map(diff_dp_annuity).astype(np.float32)

    del diff_dp_annuity
    gc.collect()

    # mean of decision on last application
    mean_last_decision = prev_app.groupby('SK_ID_CURR')['DAYS_DECISION'].mean()
    data.loc[:, 'mean_last_decision'] = data.SK_ID_CURR.map(mean_last_decision).astype(np.float32)

    del mean_last_decision
    gc.collect()

    # mean of term of previous credit
    mean_prev_credit = prev_app.groupby('SK_ID_CURR')['CNT_PAYMENT'].mean()
    data.loc[:, 'mean_prev_credit'] = data.SK_ID_CURR.map(mean_prev_credit).astype(np.float32)

    del mean_prev_credit
    gc.collect()

    # deviation in hour, weekday at which previous application process started
    dev_hour_process                = prev_app.groupby('SK_ID_CURR')['HOUR_APPR_PROCESS_START'].std()
    data.loc[:, 'dev_hour_process'] = data.SK_ID_CURR.map(dev_hour_process).astype(np.float32)

    del dev_hour_process
    gc.collect()

    dev_weekday_process                = prev_app.groupby('SK_ID_CURR')['WEEKDAY_APPR_PROCESS_START'].std()
    data.loc[:, 'dev_weekday_process'] = data.SK_ID_CURR.map(dev_weekday_process).astype(np.float32)

    del dev_weekday_process
    gc.collect()

    # mean hour, weekday at which previous application process started
    dev_hour_process                 = prev_app.groupby('SK_ID_CURR')['HOUR_APPR_PROCESS_START'].mean()
    data.loc[:, 'mean_hour_process'] = data.SK_ID_CURR.map(dev_hour_process).astype(np.float32)

    del dev_hour_process
    gc.collect()

    dev_weekday_process                 = prev_app.groupby('SK_ID_CURR')['WEEKDAY_APPR_PROCESS_START'].mean()
    data.loc[:, 'mean_weekday_process'] = data.SK_ID_CURR.map(dev_weekday_process).astype(np.float32)

    del dev_weekday_process
    gc.collect()

    # mean days before applicaiton was made
    prev_app_decision                = prev_app.groupby('SK_ID_CURR')['DAYS_DECISION'].mean()
    data.loc[:, 'prev_app_decision'] = data.SK_ID_CURR.map(prev_app_decision).astype(np.float32)

    del prev_app_decision
    gc.collect()

    # difference between termination of credit and day decision was made
    diff_termination_decision                = prev_app.DAYS_TERMINATION.replace({365243: np.nan}) - prev_app.DAYS_DECISION
    diff_termination_decision                = diff_termination_decision.groupby(prev_app.SK_ID_CURR).mean()
    data.loc[:, 'diff_termination_decision'] = data.SK_ID_CURR.map(diff_termination_decision).astype(np.float32)

    del diff_termination_decision
    gc.collect()

    # ratio of amt annuity and amt goods price
    ratio_prev_annuity_goods                = (prev_app.AMT_ANNUITY / prev_app.AMT_GOODS_PRICE).replace([np.inf, -np.inf], np.nan)
    ratio_prev_annuity_goods                = ratio_prev_annuity_goods.groupby(prev_app.SK_ID_CURR).mean()
    data.loc[:, 'mean_prev_annuity_goods']  = data.SK_ID_CURR.map(ratio_prev_annuity_goods).astype(np.float32)

    # max of ratio of amt annuity to amt goods price
    ratio_prev_annuity_goods                = (prev_app.AMT_ANNUITY / prev_app.AMT_GOODS_PRICE).replace([np.inf, -np.inf], np.nan)
    ratio_prev_annuity_goods                = ratio_prev_annuity_goods.groupby(prev_app.SK_ID_CURR).max()
    data.loc[:, 'max_prev_annuity_goods']   = data.SK_ID_CURR.map(ratio_prev_annuity_goods).astype(np.float32)

    del ratio_prev_annuity_goods
    gc.collect()


    # max of ratio of amt annuity to amt_credit_sum
    ratio_prev_annuity_credit                = (prev_app.AMT_ANNUITY / prev_app.AMT_CREDIT).replace([np.inf, -np.inf], np.nan)
    ratio_prev_annuity_credit                = ratio_prev_annuity_credit.groupby(prev_app.SK_ID_CURR).max()
    data.loc[:, 'max_prev_annuity_credit']   = data.SK_ID_CURR.map(ratio_prev_annuity_credit).astype(np.float32)

    del ratio_prev_annuity_credit
    gc.collect()


    # most recent previous application
    most_recent_prev_app = prev_app.groupby('SK_ID_CURR')['DAYS_DECISION'].max()
    data.loc[:, 'most_recent_prev_application'] = data.SK_ID_CURR.map(most_recent_prev_app).astype(np.float32)

    del most_recent_prev_app
    gc.collect()


    # credit duration
    tmp = prev_app.DAYS_TERMINATION.replace({365243.0: np.nan}) - prev_app.DAYS_DECISION.replace({365243.0: np.nan})
    tmp = tmp.groupby(prev_app.SK_ID_CURR).mean()

    data.loc[:, 'credit_duration_prev'] = data.SK_ID_CURR.map(tmp)

    del tmp
    gc.collect()

    # number of different reasons of getting rejections in previous applications
    tmp = prev_app.groupby('SK_ID_CURR')['CODE_REJECT_REASON'].nunique()
    data.loc[:, 'num_diff_reasons_rejections'] = data.SK_ID_CURR.map(tmp).fillna(-1).astype(np.int8)


    # number of running loans
    tmp                          = prev_app.DAYS_TERMINATION.map(lambda x: (x > 0) or pd.isnull(x)).astype(np.uint8)
    tmp                          = tmp.groupby(prev_app.SK_ID_CURR).sum()
    data.loc[:, 'running_loans'] = data.SK_ID_CURR.map(tmp)

    del tmp
    gc.collect()

    # ratio of annuity to credit of running loans
    res = prev_app.loc[(prev_app.DAYS_TERMINATION > 0) | (prev_app.DAYS_TERMINATION.isnull()), :]
    res = res.groupby('SK_ID_CURR')[['AMT_CREDIT', 'AMT_ANNUITY']].sum()
    res = (res.AMT_ANNUITY / res.AMT_CREDIT).replace([np.inf, -np.inf], np.nan)
    data.loc[:, 'ratio_annuity_credit_running_loans'] = data.SK_ID_CURR.map(res)

    del res
    gc.collect()

    # min of credit to goods price over different credits reported by Bureau
    res = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0) &\
             ((prev_app.DAYS_TERMINATION > 0) | (prev_app.DAYS_TERMINATION.isnull()))
             , ['SK_ID_CURR',
               'SK_ID_PREV',
               'AMT_CREDIT',
               'AMT_GOODS_PRICE'
              ]]

    tmp = (res.AMT_CREDIT / res.AMT_GOODS_PRICE).replace([np.inf, -np.inf], np.nan)
    tmp = tmp.groupby(res.SK_ID_CURR).min()
    data.loc[:, 'min_credit_goods_price_bureau'] = data.SK_ID_CURR.map(tmp)

    del res, tmp
    gc.collect()

    # number of high interest loans and loans with no information taken 
    # by person in question reported by Bureau

    res = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0) &\
                   ((prev_app.DAYS_TERMINATION > 0) | (prev_app.DAYS_TERMINATION.isnull())) &\
                   (prev_app.NAME_YIELD_GROUP.isin([0, 1]) )
             , ['SK_ID_CURR'
              ]]
    res = res.groupby(['SK_ID_CURR']).size()
    data.loc[:, 'num_high_int_no_info_loans'] = data.SK_ID_CURR.map(res).fillna(-1).astype(np.int8)

    del res
    gc.collect()

    #  difference between payment rate between approved and refused applications
    r1 = (prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, 'AMT_ANNUITY'] /\
          prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, 'AMT_CREDIT']).replace([np.inf, -np.inf], np.nan)
    r1 = r1.reset_index()
    r1.loc[:, 'index'] = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, 'SK_ID_CURR'].values
    r1 = r1.rename(columns={'index': 'SK_ID_CURR',
                            0: 'payment_rate'
                        })
    r1 = r1.groupby('SK_ID_CURR')['payment_rate'].max()

    r2 = (prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 2, 'AMT_ANNUITY'] /\
          prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 2, 'AMT_CREDIT']).replace([np.inf, -np.inf], np.nan)

    r2 = r2.reset_index()
    r2.loc[:, 'index'] = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 2, 'SK_ID_CURR'].values
    r2 = r2.rename(columns={'index': 'SK_ID_CURR',
                            0: 'payment_rate'
                        })
    r2 = r2.groupby('SK_ID_CURR')['payment_rate'].min()

    res = r1.subtract(r2, fill_value=np.nan)
    data.loc[:, 'max_approved_min_refused_payment_rate'] = data.SK_ID_CURR.map(res)

    del r1, r2, res
    gc.collect()

    return data, list(set(data.columns) - set(COLS))

def pos_cash_features(pos_cash, data):
    COLS = data.columns.tolist()
    
    data, cif_cols = get_agg_features(data, pos_cash, 'CNT_INSTALMENT_FUTURE', 'SK_ID_CURR')

    # mean of term of previous credits
    mean_term = pos_cash.groupby('SK_ID_CURR')['CNT_INSTALMENT'].mean()
    data.loc[:, 'mean_term'] = data.SK_ID_CURR.map(mean_term).astype(np.float32)

    # total number of installments
    total_installments                = pos_cash.CNT_INSTALMENT + pos_cash.CNT_INSTALMENT_FUTURE
    total_installments                = total_installments.groupby(pos_cash.SK_ID_CURR).sum()
    data.loc[:, 'total_installments'] = data.SK_ID_CURR.map(total_installments)

    # ratio of paid to unpaid number of installments
    ratio_paid_unpaid = (pos_cash.CNT_INSTALMENT_FUTURE / pos_cash.CNT_INSTALMENT).replace([np.inf, -np.inf], np.nan)
    ratio_paid_unpaid = ratio_paid_unpaid.groupby(pos_cash.SK_ID_CURR).mean()
    data.loc[:, 'ratio_paid_unpaid'] = data.SK_ID_CURR.map(ratio_paid_unpaid)

    del mean_term, total_installments, ratio_paid_unpaid
    gc.collect()

    # total number of remaining installments
    sum_remaining_installments = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)\
                                        ['CNT_INSTALMENT_FUTURE'].min()\
                                        .drop('SK_ID_PREV', axis=1)\
                                        .groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].sum()
            
    data.loc[:, 'sum_remaining_installments'] = data.SK_ID_CURR.map(sum_remaining_installments)

    del sum_remaining_installments
    gc.collect()

    return data, list(set(data.columns) - set(COLS))

def credit_card_features(credit_bal, data):
    COLS = data.columns.tolist()
    
    # mean of amount balance during previous payments
    mean_amt_balance                = credit_bal.groupby('SK_ID_CURR')['AMT_BALANCE'].mean()
    data.loc[:, 'mean_amt_balance'] = data.SK_ID_CURR.map(mean_amt_balance)

    # mean of actual credit limit
    mean_credit_limit                = credit_bal.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].mean()
    data.loc[:, 'mean_credit_limit'] = data.SK_ID_CURR.map(mean_credit_limit).astype(np.float32)

    # total paid installments on previous credit
    total_paid_installments                = credit_bal.groupby('SK_ID_CURR')['CNT_INSTALMENT_MATURE_CUM'].sum()
    data.loc[:, 'total_paid_installments'] = data.SK_ID_CURR.map(total_paid_installments)

    # mean total drawings
    mean_total_drawings                = credit_bal.groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].mean()
    data.loc[:, 'mean_total_drawings'] = data.SK_ID_CURR.map(mean_total_drawings)

    # sum of diff between balance and credit limit
    diff_bal_credit   = credit_bal.AMT_BALANCE - credit_bal.AMT_CREDIT_LIMIT_ACTUAL
    diff_bal_credit   = diff_bal_credit.groupby(credit_bal.SK_ID_CURR).sum()

    data.loc[:, 'diff_bal_credit'] = data.SK_ID_CURR.map(diff_bal_credit)

    # max balance to credit limit ratio
    mask = (credit_bal.MONTHS_BALANCE >= -6) & (credit_bal.MONTHS_BALANCE <= -1) & (credit_bal.NAME_CONTRACT_STATUS == 0)
    
    tmp  = credit_bal.loc[mask, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL']]
    res  = (tmp.AMT_BALANCE / tmp.AMT_CREDIT_LIMIT_ACTUAL).replace([np.inf, -np.inf], np.nan)
    res  = res.groupby(tmp.SK_ID_CURR).max()
    data.loc[:, 'max_bal_credit_limit'] = data.SK_ID_CURR.map(res)
    
    del res, tmp
    gc.collect()

    # difference of balance with total amount paid in a particular month
    mask = (credit_bal.MONTHS_BALANCE >= -12) & (credit_bal.MONTHS_BALANCE <= -1) & (credit_bal.NAME_CONTRACT_STATUS == 0)
    
    tmp  = credit_bal.loc[mask, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_BALANCE', 'AMT_PAYMENT_TOTAL_CURRENT']]
    res  = (tmp.AMT_BALANCE - tmp.AMT_PAYMENT_TOTAL_CURRENT)
    res  = res.groupby(tmp.SK_ID_CURR).mean()
    data.loc[:, 'mean_bal_payment_diff'] = data.SK_ID_CURR.map(res)
    
    del res, tmp
    gc.collect()

    # aggregate features for MONTHS_BALANCE
    data, mb_cols = get_agg_features(data, credit_bal, 'MONTHS_BALANCE', 'SK_ID_CURR')


    # ratio of minimum installment on credit card with amount balance
    ratio_min_installment_balance                = (credit_bal.AMT_BALANCE / credit_bal.AMT_INST_MIN_REGULARITY).replace([np.inf, -np.inf], np.nan)
    ratio_min_installment_balance                = ratio_min_installment_balance.groupby(credit_bal.SK_ID_CURR).mean()
    data.loc[:, 'ratio_min_installment_balance'] = data.SK_ID_CURR.map(ratio_min_installment_balance)

    # difference of minimum installment on credit card with amount balance
    diff_min_installment_balance                = (credit_bal.AMT_BALANCE - credit_bal.AMT_INST_MIN_REGULARITY).replace([np.inf, -np.inf], np.nan)
    diff_min_installment_balance                = diff_min_installment_balance.groupby(credit_bal.SK_ID_CURR).mean().map(lambda x: np.log(x + 1))
    data.loc[:, 'diff_min_installment_balance'] = data.SK_ID_CURR.map(diff_min_installment_balance).astype(np.float32)

    # difference between oldest credit limit and recent credit limit
    rmax = credit_bal.loc[credit_bal.NAME_CONTRACT_STATUS == 0, :].groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['MONTHS_BALANCE'].max()
    rmin = credit_bal.loc[credit_bal.NAME_CONTRACT_STATUS == 0, :].groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['MONTHS_BALANCE'].min()
    rmax = credit_bal.loc[credit_bal.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL']]\
                .merge(rmax)
    rmin = credit_bal.loc[credit_bal.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL']]\
                .merge(rmin)
        
    rmax = rmax.set_index(['SK_ID_CURR', 'SK_ID_PREV'])['AMT_CREDIT_LIMIT_ACTUAL']
    rmin = rmin.set_index(['SK_ID_CURR', 'SK_ID_PREV'])['AMT_CREDIT_LIMIT_ACTUAL']

    res  = (rmin - rmax).reset_index().drop('SK_ID_PREV', axis=1)
    res  = res.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].mean()
    data.loc[:, 'range_min_max_credit_limit'] = data.SK_ID_CURR.map(res)

    del rmax, rmin, res
    gc.collect()
    
    del mean_amt_balance, mean_credit_limit
    del total_paid_installments, mean_total_drawings, diff_bal_credit
    del ratio_min_installment_balance, diff_min_installment_balance
    gc.collect()

    return data, list(set(data.columns) - set(COLS))

def get_installment_features(installments, data):
    COLS = data.columns.tolist()

    # mean installment
    mean_installment                = installments.groupby('SK_ID_CURR')['AMT_INSTALMENT'].mean()
    data.loc[:, 'mean_installment'] = data.SK_ID_CURR.map(mean_installment)

    # mean payment against installment
    data, ap_cols   = get_agg_features(data, installments, 'AMT_PAYMENT', 'SK_ID_CURR')

    # logarithm of the features
    data  = log_features(data, ap_cols)

    # difference between actual day of installment versus when it was supposed to be paid

    diff_actual_decided = -(installments.DAYS_ENTRY_PAYMENT - installments.DAYS_INSTALMENT)
    diff_actual_decided = diff_actual_decided.groupby(installments.SK_ID_CURR).mean()

    data.loc[:, 'diff_actual_decided'] = data.SK_ID_CURR.map(diff_actual_decided)

    # ratio of installment to be paid versus actual amount paid

    res = (installments.AMT_INSTALMENT / installments.AMT_PAYMENT).replace([np.inf, -np.inf], np.nan)
    res = res.groupby(installments.SK_ID_CURR).mean()

    data.loc[:, 'ratio_actual_decided_amount'] = data.SK_ID_CURR.map(res)

    del mean_installment, diff_actual_decided, res
    gc.collect()

    # differenece between actual installment amount vs what was paid

    res = (installments.AMT_INSTALMENT - installments.AMT_PAYMENT)
    res = res.groupby(installments.SK_ID_CURR).mean()

    data.loc[:, 'diff_actual_decided_amount'] = data.SK_ID_CURR.map(res)

    del res
    gc.collect()

    return data, list(set(data.columns) - set(COLS))

def prev_curr_app_features(prev_app, data):
    COLS = data.columns.tolist()
    
    # ratio of mean of amount credit sum from previous applications and current amount credit sum
    prev_amt_credit_mean = prev_app.groupby('SK_ID_CURR')['AMT_CREDIT'].mean()
    prev_amt_credit_mean = data.SK_ID_CURR.map(prev_amt_credit_mean)

    data.loc[:, 'ratio_prev_curr_credit'] = (data.AMT_CREDIT / prev_amt_credit_mean)\
                                                .replace([np.inf, -np.inf], np.nan)


    # diff (amt_annuity / amt_credit) current and previous application
    ratio_annuity_credit = (prev_app.AMT_ANNUITY / prev_app.AMT_CREDIT).replace([np.inf, -np.inf], np.nan)
    ratio_annuity_credit = ratio_annuity_credit.groupby(prev_app.SK_ID_CURR).mean()
    data.loc[:, 'diff_annuity_credit_curr_prev'] = data.ratio_annuity_credit - ratio_annuity_credit


    # diff (amt_goods_price / amt_credit_sum) current and previous application
    ratio_goods_credit = (prev_app.AMT_GOODS_PRICE / prev_app.AMT_CREDIT).replace([np.inf, -np.inf], np.nan)
    ratio_goods_credit = ratio_goods_credit.groupby(prev_app.SK_ID_CURR).mean()
    data.loc[:, 'diff_credit_goods_curr_prev'] = data.ratio_goods_credit - ratio_goods_credit

    del prev_amt_credit_mean, ratio_annuity_credit, ratio_goods_credit
    gc.collect()

    return data, list(set(data.columns) - set(COLS))

def prev_app_bureau(prev_app, bureau, data):
    COLS = data.columns.tolist()

    # difference between day decision was made and remaining duration of CB Credit
    dcn = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].mean()
    dd  = prev_app.groupby('SK_ID_CURR')['DAYS_DECISION'].mean()
    data.loc[:, 'diff_decision_credit_end'] = data.SK_ID_CURR.map(dd - dcn).astype(np.float32)

    dcu = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].mean() 
    dd  = prev_app.groupby('SK_ID_CURR')['DAYS_DECISION'].mean()
    data.loc[:, 'diff_decision_update'] = data.SK_ID_CURR.map(dd - dcu).astype(np.float32)

    df  = bureau.groupby('SK_ID_CURR')['DAYS_ENDDATE_FACT'].mean() 
    dd  = prev_app.groupby('SK_ID_CURR')['DAYS_DECISION'].mean()
    data.loc[:, 'diff_decision_fact'] = data.SK_ID_CURR.map(dd - df).astype(np.float32)


    # difference between day at which decision for last application to Home Credit and maximum of days
    # since CB credit ended at time of application
    res = bureau.loc[:, ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_ENDDATE_FACT']]\
            .merge(prev_app.loc[:, ['SK_ID_CURR', 'DAYS_DECISION']], how='left')
    res = res.groupby('SK_ID_CURR')[['DAYS_ENDDATE_FACT', 'DAYS_DECISION']].max()
    res = res.DAYS_DECISION - res.DAYS_ENDDATE_FACT
    data.loc[:, 'diff_cb_decision_bureau_last'] = data.SK_ID_CURR.map(res).astype(np.float32)

    # diff between day of decision of previous application and credit enddate of last credit reported
    # to Bureau
    res = bureau.loc[:, ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE']]\
            .merge(prev_app.loc[:, ['SK_ID_CURR', 'DAYS_DECISION']], how='left')
    res = res.groupby('SK_ID_CURR')[['DAYS_CREDIT_ENDDATE', 'DAYS_DECISION']].max()
    res = res.DAYS_DECISION - res.DAYS_CREDIT_ENDDATE
    data.loc[:, 'diff_cb_decision_enddate'] = data.SK_ID_CURR.map(res).astype(np.float32)

    # diff between day of decision of previous application and how many days before current application client
    # applied for Credit Bureau Credit
    res = bureau.loc[:, ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']]\
            .merge(prev_app.loc[:, ['SK_ID_CURR', 'DAYS_DECISION']], how='left')
    res = res.groupby('SK_ID_CURR')[['DAYS_CREDIT', 'DAYS_DECISION']].max()
    res = res.DAYS_DECISION - res.DAYS_CREDIT
    data.loc[:, 'diff_cb_decision_credit'] = data.SK_ID_CURR.map(res).astype(np.float32)


    # diff between days termination and days credit enddate
    res = bureau.loc[:, ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE']]\
            .merge(prev_app.loc[:, ['SK_ID_CURR', 'DAYS_TERMINATION']]\
                            .replace({'DAYS_TERMINATION': {365243: np.nan}}), how='left')
    res = res.groupby('SK_ID_CURR')[['DAYS_CREDIT_ENDDATE', 'DAYS_TERMINATION']].mean()
    res = res.DAYS_TERMINATION - res.DAYS_CREDIT_ENDDATE

    data.loc[:, 'diff_termination_credit_enddate'] = data.SK_ID_CURR.map(res).astype(np.float32)

    # diff between days termination and days credit
    res = bureau.loc[:, ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']]\
            .merge(prev_app.loc[:, ['SK_ID_CURR', 'DAYS_TERMINATION']]\
                            .replace({'DAYS_TERMINATION': {365243: np.nan}}), how='left')
    res = res.groupby('SK_ID_CURR')[['DAYS_CREDIT', 'DAYS_TERMINATION']].mean()
    res = res.DAYS_TERMINATION - res.DAYS_CREDIT

    data.loc[:, 'diff_termination_credit'] = data.SK_ID_CURR.map(res).astype(np.float32)

    # diff between days termination and enddate fact
    res = bureau.loc[:, ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_ENDDATE_FACT']]\
            .merge(prev_app.loc[:, ['SK_ID_CURR', 'DAYS_TERMINATION']]\
                            .replace({'DAYS_TERMINATION': {365243: np.nan}}), how='left')
    res = res.groupby('SK_ID_CURR')[['DAYS_ENDDATE_FACT', 'DAYS_TERMINATION']].mean()
    res = res.DAYS_TERMINATION - res.DAYS_ENDDATE_FACT

    data.loc[:, 'diff_termination_enddate'] = data.SK_ID_CURR.map(res).astype(np.float32)

    del dcn, dd, dcu, df
    gc.collect()

    return data, list(set(data.columns) - set(COLS))

def prev_app_credit_card(prev_app, credit_bal, data):
    COLS = data.columns.tolist()
    
    mask = prev_app.NAME_CONTRACT_TYPE == 2

    # difference between credit amount montly balance
    res  = prev_app.loc[mask, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT']]\
                .merge(credit_bal.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_BALANCE']],
                        on=['SK_ID_CURR', 'SK_ID_PREV'],
                        how='left'
                        )
    tmp  = res.AMT_BALANCE - res.AMT_CREDIT
    tmp  = tmp.groupby(res.SK_ID_CURR).mean()
    data.loc[:, 'diff_prev_credit_balance'] = data.SK_ID_CURR.map(tmp).astype(np.float32)

    # number of times account balance was zero
    res  = prev_app.loc[mask, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT']]\
                .merge(credit_bal.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'AMT_BALANCE']],
                        on=['SK_ID_CURR', 'SK_ID_PREV'],
                        how='left'
                        )
        
    res = res.groupby('SK_ID_CURR')['AMT_BALANCE'].apply(lambda x: np.sum(x == 0))
    data.loc[:, 'num_times_balance_zero'] = data.SK_ID_CURR.map(res).astype(np.float32)

    # number of times money was withdrawn from ATM in last 6 months
    res = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR',
                                                         'SK_ID_PREV']]\
        .merge(credit_bal.loc[credit_bal.MONTHS_BALANCE > -6, 
                              ['SK_ID_CURR', 'SK_ID_PREV', 'CNT_DRAWINGS_ATM_CURRENT'
                              ]])
    res  = res.groupby('SK_ID_CURR')['CNT_DRAWINGS_ATM_CURRENT'].sum()
    data.loc[:, 'num_times_atm_withdrawn']  = data.SK_ID_CURR.map(res).fillna(-1).astype(np.int8)

    del res
    gc.collect()

    return data, list(set(data.columns) - set(COLS))

def prev_app_installments(prev_app, installments, data):
    COLS = data.columns.tolist()

    res = prev_app.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT']]\
                  .merge(installments.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_PAYMENT'
                                   ]], how='left')

    tmp = res.groupby(['SK_ID_CURR', 'SK_ID_PREV']).size().reset_index()
    tmp = tmp.groupby('SK_ID_CURR')[0].mean()

    data.loc[:, 'mean_num_installments_prev_application'] = data.SK_ID_CURR.map(tmp)

    # maximum number of installments paid in any of the previous loans
    res = prev_app.loc[:, ['SK_ID_PREV', 
                        'SK_ID_CURR', 
                        'AMT_CREDIT']]\
            .merge(installments.loc[:, ['SK_ID_PREV', 
                                        'SK_ID_CURR', 
                                        'NUM_INSTALMENT_NUMBER', 
                                        'AMT_INSTALMENT']],
                on=['SK_ID_PREV', 'SK_ID_CURR'],
                how='left'
                )
    tmp = res.groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['NUM_INSTALMENT_NUMBER'].max()\
            .groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].max()
        
    data.loc[:, 'max_num_installments_prev_credits'] = data.SK_ID_CURR.map(tmp)

    del res, tmp
    gc.collect()

    # ratio of maximum payment paid for an active loan w.r.t to income
    tmp = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR', 'SK_ID_PREV']]
    tmp = tmp.merge(installments.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_PAYMENT']], how='left')

    tmp = tmp.set_index('SK_ID_CURR')['AMT_PAYMENT']
    ss  = data.set_index('SK_ID_CURR')['AMT_INCOME_TOTAL']
    res = tmp.div(ss, fill_value=np.nan)
    res = res.reset_index()
    res = res.groupby('SK_ID_CURR')[0].max()

    data.loc[:, 'ratio_payment_income'] = data.SK_ID_CURR.map(res)

    del tmp, res, ss
    gc.collect()

    # difference between when installments were supposed to be paid and when were they actually paid
    tmp = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR', 'SK_ID_PREV']]
    tmp = tmp.merge(installments.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']], how='left')

    res = (tmp.DAYS_INSTALMENT - tmp.DAYS_ENTRY_PAYMENT)
    res = res.groupby(tmp.SK_ID_CURR).sum()
    data.loc[:, 'delay_in_installment_payments'] = data.SK_ID_CURR.map(res).replace([np.inf, -np.inf], np.nan)

    del tmp, res
    gc.collect()

    # difference between when installments were supposed to be paid and when were they actually paid
    # for applications that are still in progress.
    tmp = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0) &\
                       (prev_app.DAYS_TERMINATION > 0)
                       , ['SK_ID_CURR', 'SK_ID_PREV']]
    tmp = tmp.merge(installments.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']], how='left')

    res = (tmp.DAYS_INSTALMENT - tmp.DAYS_ENTRY_PAYMENT)
    res = res.groupby(tmp.SK_ID_CURR).sum()
    data.loc[:, 'max_delay_in_installment_payments_running'] = data.SK_ID_CURR.map(res).replace([np.inf, -np.inf], np.nan)

    del tmp, res
    gc.collect()

    # difference between  installment amount and actual paid amount
    tmp = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR', 'SK_ID_PREV']]
    tmp = tmp.merge(installments.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_INSTALMENT', 'AMT_PAYMENT']], how='left')

    res = (tmp.AMT_INSTALMENT - tmp.AMT_PAYMENT)
    res = res.groupby(tmp.SK_ID_CURR).sum()
    data.loc[:, 'delay_in_installment_amount'] = data.SK_ID_CURR.map(res).replace([np.inf, -np.inf], np.nan)

    del tmp, res
    gc.collect()

    # deviation in delay in time to amount
    tmp = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR', 'SK_ID_PREV']]
    tmp = tmp.merge(installments, how='left')

    t   = (tmp.DAYS_INSTALMENT - tmp.DAYS_ENTRY_PAYMENT)
    p   = (tmp.AMT_INSTALMENT - tmp.AMT_PAYMENT)
    res = (t - p).groupby(tmp.SK_ID_CURR).std()

    data.loc[:, 'ratio_time_amount_diff'] = data.SK_ID_CURR.map(res)

    del res, tmp, t, p
    gc.collect()

    # number of times repayment amount was less than installment loan
    tmp = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR', 'SK_ID_PREV']]
    tmp = tmp.merge(installments.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_INSTALMENT', 'AMT_PAYMENT']], how='left')

    tmp.loc[:, 'diff_inst_payment'] = (tmp.AMT_INSTALMENT - tmp.AMT_PAYMENT).astype(np.float32)
    tmp.loc[:, 'neg_inst_payment']  = (tmp.diff_inst_payment > 0).astype(np.int8)

    res = tmp.groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['neg_inst_payment'].sum().drop('SK_ID_PREV', axis=1)
    res = res.groupby('SK_ID_CURR')['neg_inst_payment'].sum()
    data.loc[:, 'num_times_le_repayment'] = data.SK_ID_CURR.map(res).fillna(-1).astype(np.int8)

    del res, tmp
    gc.collect()

    return data, list(set(data.columns) - set(COLS))

def loan_stacking(bureau, prev_app, credit_bal, data):
    COLS = data.columns.tolist()

    # find all active records and take mean of days credit
    res = bureau.loc[bureau.CREDIT_ACTIVE == 0, ['SK_ID_CURR', 'DAYS_CREDIT']]
    res = res.groupby('SK_ID_CURR')['DAYS_CREDIT'].mean()

    data.loc[:, 'stacked_loan_credit_start'] = data.SK_ID_CURR.map(res)

    del res
    gc.collect()

    # find all active records and take sum of amount credited
    res = bureau.loc[bureau.CREDIT_ACTIVE == 0, ['SK_ID_CURR', 'AMT_CREDIT_SUM']]
    res = res.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum()

    data.loc[:, 'stacked_loan_credit_sum'] = data.SK_ID_CURR.map(res)
    
    del res
    gc.collect()

    # when was last update about active credit reached bureau ?
    res = bureau.loc[bureau.CREDIT_ACTIVE == 0, ['SK_ID_CURR', 'DAYS_CREDIT_UPDATE']]
    res = -res.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].max()

    data.loc[:, 'last_update_active_records'] = data.SK_ID_CURR.map(res).fillna(-1).astype(np.int8)

    del res
    gc.collect()
    
    # number of active loan records ( for last 1 year )
    res = bureau.loc[(bureau.CREDIT_ACTIVE == 0) & (bureau.DAYS_CREDIT > -365*1), ['SK_ID_CURR', 'DAYS_CREDIT_UPDATE']]
    res = res.groupby('SK_ID_CURR').size()

    data.loc[:, 'num_active_loan_records'] = data.SK_ID_CURR.map(res).fillna(-1).astype(np.int8)

    del res
    gc.collect()

    # total amount credit courtesy of recent loans ( for last 1 year )
    res = bureau.loc[(bureau.CREDIT_ACTIVE == 0) & (bureau.DAYS_CREDIT > -365*1), ['SK_ID_CURR', 'AMT_CREDIT_SUM']]
    res = res.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum()

    res = data.SK_ID_CURR.map(res)
    data.loc[:, 'diff_active_credit_income'] = data.AMT_INCOME_TOTAL - (res + data.AMT_CREDIT)


    # repayment ability of a loanee
    res           = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0) &\
                                 (prev_app.DAYS_TERMINATION.isin([365243, np.nan])), 
                                 ['SK_ID_CURR', 'AMT_ANNUITY', 'AMT_CREDIT']]

    total_credit  = res.groupby('SK_ID_CURR')['AMT_CREDIT'].sum()
    total_annuity = res.groupby('SK_ID_CURR')['AMT_ANNUITY'].sum()
    
    r1 = data.SK_ID_CURR.map(total_annuity)
    r2 = data.SK_ID_CURR.map(total_credit)

    data.loc[:, 'total_annuity_to_credit_hc']      = (r1 + data.AMT_ANNUITY) / (r2 + data.AMT_CREDIT)
    data.loc[:, 'annuity_to_total_income']         = (r1 + data.AMT_ANNUITY) / (data.AMT_INCOME_TOTAL) 
    data.loc[:, 'total_income_total_diff_annuity'] = (data.AMT_INCOME_TOTAL) - (r1 + data.AMT_ANNUITY)

    del r1, r2, res
    gc.collect()

    # adjust total income by number of family members as well
    data.loc[:, 'adjusted_total_income'] = (data.AMT_INCOME_TOTAL / data.CNT_FAM_MEMBERS)

    # most recent balance during active credits and take difference with current amount credited
    res = credit_bal.loc[(credit_bal.MONTHS_BALANCE == -1) & (credit_bal.NAME_CONTRACT_STATUS == 0), :]\
                .groupby('SK_ID_CURR')['AMT_BALANCE'].mean()
    
    res = data.SK_ID_CURR.map(res)
    data.loc[:, 'diff_balance_curr_credit'] = res - data.AMT_CREDIT 

    del res
    gc.collect()

    # number of times previous application was rejected.
    res = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 2), :].groupby('SK_ID_CURR').size()
    data.loc[:, 'num_applications_refused'] = data.SK_ID_CURR.map(res).fillna(-1).astype(np.int8)

    del res
    gc.collect()

    # merge information about ownership of car or house
    data.loc[:, 'own_house_car'] = pd.factorize(data.FLAG_OWN_CAR.astype(np.str) + '_' + data.FLAG_OWN_REALTY.astype(np.str))[0]
    data.loc[:, 'own_house_car'] = data.own_house_car.astype(np.int8)
    
    # merge information about income and education status
    data.loc[:, 'income_education'] = pd.factorize(data.NAME_INCOME_TYPE.astype(np.str) + '_' + data.NAME_EDUCATION_TYPE.astype(np.str))[0]
    data.loc[:, 'income_education'] = data.income_education.astype(np.int8)

    # merge housing and family type
    data.loc[:, 'family_housing'] = pd.factorize(data.NAME_FAMILY_STATUS.astype(np.str) + '_' + data.NAME_HOUSING_TYPE.astype(np.str))[0]
    data.loc[:, 'family_housing'] = data.family_housing.astype(np.int8)

    # merge income, education, family and housing
    data.loc[:, 'income_edu_fam_housing'] = pd.factorize(data.NAME_INCOME_TYPE.astype(np.str) + '_' +\
                                                         data.NAME_EDUCATION_TYPE.astype(np.str) + '_' +\
                                                         data.NAME_FAMILY_STATUS.astype(np.str) + '_' +\
                                                         data.NAME_HOUSING_TYPE.astype(np.str)
                                                        )[0]
    data.loc[:, 'income_edu_fam_housing'] = data.income_edu_fam_housing.astype(np.int8)

    # string representation of documents provided by client for current application
    data.loc[:, 'documents_provided'] = data.loc[:, [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]]\
                                            .apply(lambda x: ''.join(x.astype(np.str)), axis=1)
    data.loc[:, 'documents_provided'] = pd.factorize(data.documents_provided)[0]
    data.loc[:, 'documents_provided'] = data.documents_provided.astype(np.int8)

    # string representation of phone, email information
    data.loc[:, 'phone_email_info']   = data.loc[:, ['FLAG_MOBIL',
                                                     'FLAG_EMP_PHONE',
                                                     'FLAG_WORK_PHONE',
                                                     'FLAG_CONT_MOBILE',
                                                     'FLAG_PHONE',
                                                     'FLAG_EMAIL'
                                                    ]].apply(lambda x: ''.join(x.astype(np.str)), axis=1)
    data.loc[:, 'phone_email_info']   = pd.factorize(data.phone_email_info)[0]
    data.loc[:, 'phone_email_info']   = data.phone_email_info.astype(np.int8)

    # check how many details are not presented correctly
    data.loc[:, 'num_times_region_false_info'] = data.loc[:, ['REG_REGION_NOT_LIVE_REGION',
                                                              'REG_REGION_NOT_WORK_REGION',
                                                              'LIVE_REGION_NOT_WORK_REGION',
                                                              'REG_CITY_NOT_LIVE_CITY',
                                                              'REG_CITY_NOT_WORK_CITY',
                                                              'LIVE_CITY_NOT_WORK_CITY'
                                                             ]].apply(np.sum, axis=1)
    data.loc[:, 'num_times_region_false_info'] = data.num_times_region_false_info.astype(np.int8)

    # feature group statistics on some feature groups
    data.loc[:, 'fg_avg'] = data.loc[:, [
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
                                        'NONLIVINGAREA_AVG'
                                    ]].apply(np.mean, axis=1)

    data.loc[:, 'fg_mode'] = data.loc[:, [
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
                                        'TOTALAREA_MODE'
                                    ]].apply(np.mean, axis=1)


    return data, list(set(data.columns) - set(COLS))

def feature_groups(data):
    COLS = data.columns.tolist()

    # number of observations of surroundings exceeding day limit
    data.loc[:, 'surroundings_past_by_mean'] = data.loc[:, ['OBS_30_CNT_SOCIAL_CIRCLE',
                                                            'DEF_30_CNT_SOCIAL_CIRCLE',
                                                            'OBS_60_CNT_SOCIAL_CIRCLE',
                                                            'DEF_60_CNT_SOCIAL_CIRCLE']].apply(np.mean, axis=1)


    # mean information about client surroundings
    data.loc[:, 'fg_medi']  = data.loc[:, ['APARTMENTS_MEDI',
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
                                            'NONLIVINGAREA_MEDI'
                                        ]].apply(np.mean, axis=1)

    # summarize information regarding queries to Home Credit about client.
    data.loc[:, 'fg_enquiries'] = data.loc[:, ['AMT_REQ_CREDIT_BUREAU_HOUR',
                                               'AMT_REQ_CREDIT_BUREAU_DAY',
                                               'AMT_REQ_CREDIT_BUREAU_WEEK',
                                               'AMT_REQ_CREDIT_BUREAU_MON',
                                               'AMT_REQ_CREDIT_BUREAU_QRT',
                                               'AMT_REQ_CREDIT_BUREAU_YEAR'
                                               ]].apply(np.mean, axis=1)

    # merge hour, weekday at which process was started
    data.loc[:, 'hour_weekday_curr_app']  = data.HOUR_APPR_PROCESS_START.astype(np.str) + '_' +\
                                            data.WEEKDAY_APPR_PROCESS_START.astype(np.str)

    data.loc[:, 'hour_weekday_curr_app']  = pd.factorize(data.hour_weekday_curr_app)[0]
    data.loc[:, 'hour_weekday_curr_app']  = data.hour_weekday_curr_app.astype(np.int8)


    return data, list(set(data.columns) - set(COLS)) 

def prev_app_pos(prev_app, pos_cash, data):
    COLS = data.columns.tolist()

    # all active cash loans
    tmp = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR', 'SK_ID_PREV']]
    tmp = tmp.merge(pos_cash.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'CNT_INSTALMENT_FUTURE']], how='left')

    res = tmp.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].sum()
    data.loc[:, 'total_remaining_cash_credit_term'] = data.SK_ID_CURR.map(res)

    del res, tmp
    gc.collect()

    return data, list(set(data.columns) - set(COLS)) 