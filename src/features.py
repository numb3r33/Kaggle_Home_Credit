import pandas as pd
import numpy as np

import gc

def frequency_encoding(data, cols):
    for col in cols:
        data.loc[:, col] = data.groupby(col)[col].transform(lambda x: len(x)).values
    return data

def one_hot_encoding(data, cols, drop_col=True):
    for col in cols:
        ohe_df = pd.get_dummies(data[col].astype(np.str), dummy_na=True, prefix=f'{col}_')

        # drop the column passed
        if drop_col:
            data.drop(col, axis=1, inplace=True)
        
        data = pd.concat((data, ohe_df), axis=1)

    return data

def merge(data, tmp):
    data = data.merge(tmp, on='SK_ID_CURR', how='left')
    cols = list(tmp.columns.drop('SK_ID_CURR'))
    data.loc[:, cols] = data[cols].fillna(-999).astype(np.int16)

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

    # Alter EXT_SOURCE_1 based on Income Type ( Pensioners )
    data.loc[data.NAME_INCOME_TYPE == 3, 'EXT_SOURCE_1'] = np.nan

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

    # geometric mean
    data.loc[:, 'EXT_1_2_gm'] = np.power(data.loc[:, 'EXT_SOURCE_1'] * data.loc[:, 'EXT_SOURCE_2'], 1 / 2)
    data.loc[:, 'EXT_1_3_gm'] = np.power(data.loc[:, 'EXT_SOURCE_1'] * data.loc[:, 'EXT_SOURCE_3'], 1 / 2)
    data.loc[:, 'EXT_2_3_gm'] = np.power(data.loc[:, 'EXT_SOURCE_2'] * data.loc[:, 'EXT_SOURCE_3'], 1 / 2)
    data.loc[:, 'EXT_1_2_3_gm'] = np.power(data.loc[:, 'EXT_SOURCE_1'] * data.loc[:, 'EXT_SOURCE_2'] * data.loc[:, 'EXT_SOURCE_3'], 1 / 3)

    data.loc[:, 'EXT_1_2_sum'] = data.loc[:, 'EXT_SOURCE_1'] + data.loc[:, 'EXT_SOURCE_2']
    data.loc[:, 'EXT_1_3_sum'] = data.loc[:, 'EXT_SOURCE_1'] + data.loc[:, 'EXT_SOURCE_3']
    data.loc[:, 'EXT_2_3_sum'] = data.loc[:, 'EXT_SOURCE_2'] + data.loc[:, 'EXT_SOURCE_3']

    data.loc[:, 'EXT_1_2_mean'] = (data.loc[:, 'EXT_SOURCE_1'] + data.loc[:, 'EXT_SOURCE_2']) / 2
    data.loc[:, 'EXT_2_3_mean'] = (data.loc[:, 'EXT_SOURCE_2'] + data.loc[:, 'EXT_SOURCE_3']) / 2
    data.loc[:, 'EXT_1_3_mean'] = (data.loc[:, 'EXT_SOURCE_1'] + data.loc[:, 'EXT_SOURCE_3']) / 2
    
    data.loc[:, 'EXT_1_2_div'] = data.loc[:, 'EXT_SOURCE_1'] / data.loc[:, 'EXT_SOURCE_2']
    data.loc[:, 'EXT_1_3_div'] = data.loc[:, 'EXT_SOURCE_1'] / data.loc[:, 'EXT_SOURCE_3']
    data.loc[:, 'EXT_2_3_div'] = data.loc[:, 'EXT_SOURCE_2'] / data.loc[:, 'EXT_SOURCE_3']

    # Weighted combination of external mean scores
    weights = [.4, 3, 1.2]
    r1 = (data.EXT_SOURCE_1 * weights[0]).add(data.EXT_SOURCE_2 * weights[1], fill_value=0)
    r2 = r1.add(data.EXT_SOURCE_3 * weights[2], fill_value=0)
    data.loc[:, 'weighted_mean_external_scores'] = r2

    # nan median
    data.loc[:, 'external_scores_nan_median'] = np.nanmedian(data.loc[:, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)   

    # recent employment
    data.loc[:, 'recent_employment'] = (data['DAYS_EMPLOYED'] < -2000).astype(np.uint8)
    data.loc[:, 'young_age']         = (data['DAYS_BIRTH'] < -14000).astype(np.uint8)

    FEATURE_NAMES += ['EXT_3_2', 
                      'EXT_1_3', 
                      'EXT_1_2',
                      'EXT_1_3_gm',
                      'EXT_2_3_gm',
                      'EXT_1_2_3_gm',
                      'EXT_1_2_sum', 
                      'EXT_1_3_sum', 
                      'EXT_2_3_sum',
                      'EXT_1_2_div', 
                      'EXT_1_3_div', 
                      'EXT_2_3_div',
                      'EXT_1_2_mean',
                      'EXT_2_3_mean',
                      'EXT_1_3_mean',
                      'weighted_mean_external_scores',
                      'external_scores_nan_median',
                      'recent_employment',
                      'young_age'
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
    data.loc[:, 'diff_credit_income']  = data.loc[:, 'AMT_CREDIT'] - data.loc[:, 'AMT_INCOME_TOTAL']

    # relationship between annual amount to be paid and income
    data.loc[:, 'ratio_annuity_income'] = data.loc[:, 'AMT_ANNUITY'] / data.loc[:, 'AMT_INCOME_TOTAL']

    # relationship between amount annuity and age
    data.loc[:, 'ratio_annuity_age'] = (data.loc[:, 'AMT_ANNUITY'] / (-data.loc[:, 'DAYS_BIRTH'] / 365)).astype(np.float32)
    FEATURE_NAMES += ['ratio_credit_income', 
                      'diff_credit_income',
                      'ratio_annuity_income', 
                      'ratio_annuity_age']
    
    # number of missing values in an application
    data.loc[:, 'num_missing_values'] = data.loc[:, data.columns.drop('TARGET')].isnull().sum(axis=1).values
    data.loc[:, 'num_nulls_freq']     = data.groupby('num_missing_values')['num_missing_values'].transform(lambda x: len(x))
    mean_ext_source_2_by_num_nulls    = data.groupby('num_nulls_freq')['EXT_SOURCE_2'].mean()
    mean_ext_source_3_by_num_nulls    = data.groupby('num_nulls_freq')['EXT_SOURCE_3'].mean()

    data.loc[:, 'mean_EXT_SOURCE_2_num_nulls'] = data.num_nulls_freq.map(mean_ext_source_2_by_num_nulls)
    data.loc[:, 'mean_EXT_SOURCE_2_num_nulls'] = data.num_nulls_freq.map(mean_ext_source_3_by_num_nulls)
    
    FEATURE_NAMES += ['num_missing_values', 'num_nulls_freq', 'mean_EXT_SOURCE_2_num_nulls']
    
    # feature interaction between age and days employed
    data.loc[:, 'age_plus_employed']  = data.loc[:, 'DAYS_BIRTH'] + data.loc[:, 'DAYS_EMPLOYED']
    data.loc[:, 'ratio_age_employed'] = ((data.DAYS_EMPLOYED.replace({365243: np.nan})) / (data.DAYS_BIRTH)).astype(np.float32)
    FEATURE_NAMES += ['age_plus_employed', 'ratio_age_employed']

    # convert age, days employed into categorical variable and then concatenate those two.
    age_categorical = pd.cut(-data.DAYS_BIRTH / 365, bins=20)
    emp_categorical = pd.cut((-data.DAYS_EMPLOYED.replace({365243: np.nan}) / 365), bins=15)

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

    # children ratio
    data.loc[:, 'children_ratio'] = data.loc[:, 'CNT_CHILDREN'] / data.loc[:, 'CNT_FAM_MEMBERS']
    FEATURE_NAMES += ['children_ratio']

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
    data.loc[:, 'ratio_annuity_credit'] = data.loc[:, 'AMT_ANNUITY'] / data.loc[:, 'AMT_CREDIT'].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    data.loc[:, 'diff_annuity_credit']  = data.loc[:, 'AMT_CREDIT'] - data.loc[:, 'AMT_ANNUITY']

    # feature interaction between amount credit and age
    data.loc[:, 'ratio_credit_age'] = (data.AMT_CREDIT / (-data.DAYS_BIRTH / 365)).astype(np.float32)

    # feature interaction between amount credit and days before application id was changed
    data.loc[:, 'ratio_credit_id_change'] = (data.AMT_CREDIT / -data.DAYS_ID_PUBLISH).replace([np.inf, -np.inf], np.nan)

    # feature interaction between days id publish and age
    data.loc[:, 'ratio_id_change_age'] = (data.DAYS_ID_PUBLISH / data.DAYS_BIRTH).astype(np.float32)
    data.loc[:, 'diff_id_change_age']  = (data.DAYS_ID_PUBLISH - data.DAYS_BIRTH).astype(np.float32)
    data.loc[:, 'ratio_reg_age']       = (data.DAYS_REGISTRATION / data.DAYS_BIRTH).astype(np.float32)
    data.loc[:, 'diff_reg_age']        = (data.DAYS_REGISTRATION - data.DAYS_BIRTH).astype(np.float32)

    FEATURE_NAMES += ['ratio_annuity_credit', 
                      'diff_annuity_credit', 
                      'ratio_credit_age', 
                      'ratio_credit_id_change', 
                      'ratio_id_change_age',
                      'diff_id_change_age',
                      'ratio_reg_age',
                      'diff_reg_age'
                    ]


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
    data.loc[:, 'car_to_employ_ratio']  = (data.OWN_CAR_AGE / data.DAYS_EMPLOYED)

    # difference of car age with age
    data.loc[:, 'diff_car_age']         = (-data.DAYS_BIRTH / 365) - data.OWN_CAR_AGE
    
    # difference income total and annuity
    data.loc[:, 'diff_income_annuity']  = data.AMT_ANNUITY - data.AMT_INCOME_TOTAL

    # difference credit and goods price
    data.loc[:, 'diff_credit_goods'] = data.AMT_CREDIT - data.AMT_GOODS_PRICE
    FEATURE_NAMES += ['ratio_car_person_age', 'car_to_employ_ratio', 'diff_income_annuity', 'diff_credit_goods']
    

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

    # income credit percentage
    data['income_credit_percentage'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
    data['income_per_child']      = data['AMT_INCOME_TOTAL'] / (1 + data['CNT_CHILDREN'])
    data['income_per_person']     = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
    data['payment_rate']          = data['AMT_ANNUITY'] / data['AMT_CREDIT']
    data['phone_to_birth_ratio']  = data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_BIRTH']
    data['phone_to_employ_ratio'] = data['DAYS_LAST_PHONE_CHANGE'] / data['DAYS_EMPLOYED']
    FEATURE_NAMES += ['income_credit_percentage', 'income_per_child', 'income_per_person',
                      'payment_rate', 'phone_to_birth_ratio', 'phone_to_employ_ratio'
                     ]

    
    # relationship of monthly amount paid, annual income and external source 2
    data.loc[:, 'annuity_div_income_ext_source_2'] = ((data.AMT_ANNUITY * 12) / data.AMT_INCOME_TOTAL) * data.EXT_SOURCE_2
    data.loc[:, 'annuity_sub_income_ext_source_2'] = ((data.AMT_ANNUITY * 12) - data.AMT_INCOME_TOTAL) * data.EXT_SOURCE_2
    
    # relationship of monthly amount paid, credit amount and external source 2
    data.loc[:, 'annuity_div_credit_ext_source_2'] = ((data.AMT_ANNUITY * 12) / data.AMT_CREDIT) * data.EXT_SOURCE_2
    data.loc[:, 'annuity_sub_credit_ext_source_2'] = ((data.AMT_ANNUITY * 12) - data.AMT_CREDIT) * data.EXT_SOURCE_2

    FEATURE_NAMES += ['annuity_div_income_ext_source_2',
                      'annuity_sub_income_ext_source_2',
                      'annuity_div_credit_ext_source_2',
                      'annuity_sub_credit_ext_source_2'
                     ]

    # merge social indicators
    social_indicator = pd.factorize(data.NAME_EDUCATION_TYPE.astype(np.str) + '_' +\
                       data.NAME_FAMILY_STATUS.astype(np.str) + '_' +\
                       data.NAME_HOUSING_TYPE.astype(np.str) + '_' +\
                       data.NAME_INCOME_TYPE.astype(np.str))[0]
    social_indicator = pd.Series(social_indicator)
    social_indicator_counts   = social_indicator.value_counts()
    low_freq_social_indicator = social_indicator_counts[social_indicator_counts <= 8].index

    social_indicator[social_indicator.isin(low_freq_social_indicator)] = -99
    data.loc[:, 'social_indicator'] = social_indicator

    FEATURE_NAMES += ['social_indicator']

    # family related features
    data['cnt_non_child'] = data['CNT_FAM_MEMBERS'] - data['CNT_CHILDREN']
    data['child_to_non_child_ratio'] = data['CNT_CHILDREN'] / data['cnt_non_child']
    data['income_per_non_child'] = data['AMT_INCOME_TOTAL'] / data['cnt_non_child']
    data['credit_per_person'] = data['AMT_CREDIT'] / data['CNT_FAM_MEMBERS']
    data['credit_per_child'] = data['AMT_CREDIT'] / (1 + data['CNT_CHILDREN'])
    data['credit_per_non_child'] = data['AMT_CREDIT'] / data['cnt_non_child']

    FEATURE_NAMES += ['cnt_non_child',
                      'child_to_non_child_ratio',
                      'income_per_non_child',
                      'credit_per_person',
                      'credit_per_child',
                      'credit_per_non_child'
                     ]

    # relationship between age and external sources
    data.loc[:, 'mult_age_ext_source_1'] = (-data.DAYS_BIRTH / 365) * data.EXT_SOURCE_1
    data.loc[:, 'mult_age_ext_source_2'] = (-data.DAYS_BIRTH / 365) * data.EXT_SOURCE_2
    data.loc[:, 'mult_age_ext_source_3'] = (-data.DAYS_BIRTH / 365) * data.EXT_SOURCE_3

    data.loc[:, 'div_age_ext_source_1']  = (-data.DAYS_BIRTH / 365) / data.EXT_SOURCE_1
    data.loc[:, 'div_age_ext_source_2']  = (-data.DAYS_BIRTH / 365) / data.EXT_SOURCE_2
    data.loc[:, 'div_age_ext_source_3']  = (-data.DAYS_BIRTH / 365) / data.EXT_SOURCE_3

    FEATURE_NAMES += ['mult_age_ext_source_1',
                      'mult_age_ext_source_2',
                      'mult_age_ext_source_3',
                      'div_age_ext_source_1',
                      'div_age_ext_source_2',
                      'div_age_ext_source_3'
                     ]

    # nulls in external scores
    data.loc[:, 'null_scores_ext_scores'] = pd.factorize(data.EXT_SOURCE_1.isnull().astype(np.str) + '_' +\
                                                         data.EXT_SOURCE_2.isnull().astype(np.str) + '_' +\
                                                         data.EXT_SOURCE_3.isnull().astype(np.str))[0]
                     

    return data, FEATURE_NAMES

def bureau_features(bureau, data):
    COLS = data.columns.tolist()

    # number of previous loans for a particular user
    prev_num_loans = bureau.groupby('SK_ID_CURR').size()

    # aggregation features
    res = bureau[bureau.CREDIT_ACTIVE == 0].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].min()
    data.loc[:, 'AMT_CREDIT_SUM_min'] = data.SK_ID_CURR.map(res)

    res = bureau[bureau.CREDIT_ACTIVE == 0].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].max()
    data.loc[:, 'AMT_CREDIT_SUM_max'] = data.SK_ID_CURR.map(res)

    data.loc[:, 'ratio_max_min_amt_credit_sum'] = data.AMT_CREDIT_SUM_max.div(data.AMT_CREDIT_SUM_min, fill_value=np.nan)

    
    # number of credits completed in current year
    gp = (-bureau.loc[bureau.CREDIT_ACTIVE == 2, 'DAYS_CREDIT'] / 365).map(np.ceil)
    num_closed_loans_recent_years = bureau.loc[bureau.CREDIT_ACTIVE == 2]\
                                      .groupby(['SK_ID_CURR', gp])\
                                      .size().unstack().fillna(0)
    
    r2 = data.loc[:, ['SK_ID_CURR']].merge(num_closed_loans_recent_years, 
                                           left_on='SK_ID_CURR', 
                                           right_index=True, 
                                           how='left'
                                          ).drop('SK_ID_CURR', axis=1).fillna(0)
    data.loc[:, 'num_closed_loans_recent_years'] = r2[1].values

    del gp, r2, num_closed_loans_recent_years
    gc.collect()

    
    # mean number of days of CB credit at the time of application
    mean_days_credit_end = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].mean()

    # mean of maximum amount overdue on any credit line
    mean_max_amt_overdue = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].mean().map(lambda x: np.log(x + 1)).astype(np.float32)

    # mean of total amount overdue on any credit line
    mean_total_amt_overdue = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].mean().map(lambda x: np.log(x + 1)).astype(np.float32)
 
    # median of amount credit sum limit
    median_amt_credit_sum_limit = bureau.loc[bureau.CREDIT_ACTIVE == 0].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_LIMIT'].median()
    data.loc[:, 'median_amt_credit_sum_limit'] = data.SK_ID_CURR.map(median_amt_credit_sum_limit)

    # variance of amount credit sum limit
    var_amt_credit_sum_limit   = bureau.loc[bureau.CREDIT_ACTIVE == 0].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_LIMIT'].median()
    data.loc[:, 'var_amt_credit_sum_limit']  = data.SK_ID_CURR.map(var_amt_credit_sum_limit)

    del median_amt_credit_sum_limit, var_amt_credit_sum_limit
    gc.collect()

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
    latest_credit = bureau[bureau.CREDIT_ACTIVE == 0].groupby('SK_ID_CURR')['DAYS_CREDIT'].max()
    data.loc[:, 'latest_credit'] = data.SK_ID_CURR.map(latest_credit).astype(np.float32)

    # credit duration
    res = bureau.loc[(bureau.CREDIT_ACTIVE == 0) &\
                     (bureau.DAYS_CREDIT_ENDDATE.notnull())
                    , ['SK_ID_CURR', 'SK_ID_BUREAU', 
                        'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 
                        'AMT_CREDIT_SUM']]

    res.loc[:, 'duration'] = (res.DAYS_CREDIT_ENDDATE - res.DAYS_CREDIT).astype(np.int32)
    res.loc[:, 'sum_to_duration'] = res.AMT_CREDIT_SUM / res.duration

    tmp = res.groupby('SK_ID_CURR')['sum_to_duration'].min()
    res = res.groupby('SK_ID_CURR')['duration'].median()
    data.loc[:, 'credit_duration'] = data.SK_ID_CURR.map(res)
    data.loc[:, 'sum_to_duration'] = data.SK_ID_CURR.map(tmp)

    del res, tmp
    gc.collect()

    # duration of close credits taken from Home Credit
    res  = bureau.loc[(bureau.CREDIT_ACTIVE == 2) &\
                      (bureau.DAYS_CREDIT_ENDDATE < 0)
                      , ['SK_ID_CURR', 'DAYS_CREDIT', 'DAYS_ENDDATE_FACT']]
    d2   = -(res.DAYS_CREDIT - res.DAYS_ENDDATE_FACT).astype(np.float32)

    d2   = d2.groupby(res.SK_ID_CURR).mean().astype(np.float32)
    data.loc[:, 'closed_credit_duration']  = data.SK_ID_CURR.map(d2).astype(np.float32)

    del d2, res
    gc.collect()
    
    # ratio of active to closed loan duration
    data.loc[:, 'div_deltas']  = data.credit_duration.div(data.closed_credit_duration, fill_value=np.nan)
    data.loc[:, 'diff_deltas'] = data.credit_duration - data.closed_credit_duration

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

    # number of credits ended in last 365 days reported by Bureau
    res = bureau.loc[(bureau.CREDIT_ACTIVE == 2) &\
           (bureau.DAYS_ENDDATE_FACT > -365)
          ].groupby('SK_ID_CURR').size()
    data.loc[:, 'credits_ended_bureau'] = data.SK_ID_CURR.map(res).fillna(-1).astype(np.int8)

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
    tmp_diff    = total_sum.subtract(total_debt, fill_value=0)

    data.loc[:, 'total_debt']            = data.SK_ID_CURR.map(total_debt)
    data.loc[:, 'total_credit_sum']      = data.SK_ID_CURR.map(total_sum)
    data.loc[:, 'ratio_debt_credit_sum'] = data.SK_ID_CURR.map(tmp)
    data.loc[:, 'diff_debt_credit_sum']  = data.SK_ID_CURR.map(tmp_diff)

    del res, tmp, total_sum, total_debt, tmp_diff
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
    data.loc[:, 'mean_days_credit_end']     = data.SK_ID_CURR.map(mean_days_credit_end).fillna(0).values
    data.loc[:, 'mean_max_amt_overdue']     = data.SK_ID_CURR.map(mean_max_amt_overdue).fillna(0).values
    data.loc[:, 'mean_total_amt_overdue']   = data.SK_ID_CURR.map(mean_total_amt_overdue).values
    data.loc[:, 'sum_num_times_prolonged']  = data.SK_ID_CURR.map(sum_num_times_prolonged).fillna(0).astype(np.int8).values
    data.loc[:, 'mean_cb_credit_annuity']   = data.SK_ID_CURR.map(mean_cb_credit_annuity).fillna(0).values
    data.loc[:, 'std_cb_credit_annuity']    = data.SK_ID_CURR.map(std_cb_credit_annuity).fillna(0).values
    data.loc[:, 'num_diff_credits']         = data.SK_ID_CURR.map(num_diff_credits).fillna(0).values
    data.loc[:, 'mean_days_credit_update']  = data.SK_ID_CURR.map(mean_days_credit_update).fillna(0).values

    del prev_num_loans, mean_days_credit_end
    del mean_max_amt_overdue, mean_total_amt_overdue, sum_num_times_prolonged
    del mean_cb_credit_annuity, std_cb_credit_annuity, num_diff_credits
    del mean_days_credit_update

    gc.collect()

    # number of active loans reported to Home Credit for a person in last n number of days
    res = bureau.loc[(bureau.CREDIT_ACTIVE == 0) &\
           (bureau.DAYS_CREDIT > -(366 * 2))
           , ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']]\
            .groupby('SK_ID_CURR').size()
    
    data.loc[:, 'recent_bureau_loans'] = data.SK_ID_CURR.map(res).fillna(-1).astype(np.int8)

    del res
    gc.collect()

    # comparison of oldest loan with employment status
    oldest_credit = bureau.loc[(bureau.CREDIT_ACTIVE == 0) & (bureau.DAYS_ENDDATE_FACT.isnull()), :].groupby('SK_ID_CURR')['DAYS_CREDIT'].min()
    oldest_credit = data.SK_ID_CURR.map(oldest_credit)
    data.loc[:, 'oldest_loan_employment'] = oldest_credit - data.DAYS_EMPLOYED.replace({365243: np.nan})

    # comparison of oldest loan with age status
    oldest_credit = -bureau.loc[(bureau.CREDIT_ACTIVE == 0) & (bureau.DAYS_ENDDATE_FACT.isnull()), :].groupby('SK_ID_CURR')['DAYS_CREDIT'].min() / 365
    oldest_credit = data.SK_ID_CURR.map(oldest_credit)
    data.loc[:, 'oldest_loan_age'] = (-data.DAYS_BIRTH / 365).subtract(oldest_credit, fill_value=0)

    del oldest_credit
    gc.collect()

    # difference in number of loans taken from Credit Bureau in current year
    # from previous year
    curr_year = bureau.loc[(bureau.CREDIT_ACTIVE == 0) &\
                       (bureau.DAYS_CREDIT >= -365)
                      ].groupby('SK_ID_CURR').size()

    prev_year = bureau.loc[(bureau.CREDIT_ACTIVE == 0) &\
                        (bureau.DAYS_CREDIT < -365) &\
                        (bureau.DAYS_CREDIT >= -365*2)
                        ].groupby('SK_ID_CURR').size()

    diff_curr_prev = curr_year.subtract(prev_year, fill_value=0)
    data.loc[:, 'diff_curr_prev_num_credits'] = data.SK_ID_CURR.map(diff_curr_prev).fillna(-254).astype(np.int8)

    del curr_year, prev_year, diff_curr_prev
    gc.collect()

    res = (-bureau.loc[bureau.DAYS_ENDDATE_FACT.isnull(), :].groupby('SK_ID_CURR')['DAYS_CREDIT'].max() / 365)
    res = data.SK_ID_CURR.map(res)

    data.loc[:, 'max_loan_age'] = (-data.DAYS_BIRTH / 365).subtract(res, fill_value=0)
    
    res = (-bureau.loc[bureau.DAYS_ENDDATE_FACT.isnull(), :].groupby('SK_ID_CURR')['DAYS_CREDIT'].median() / 365)
    res = data.SK_ID_CURR.map(res)

    data.loc[:, 'median_loan_age'] = (-data.DAYS_BIRTH / 365).subtract(res, fill_value=0)
    
    res = bureau.loc[bureau.CREDIT_ACTIVE == 2, ['SK_ID_CURR', 'DAYS_ENDDATE_FACT']]
    res = res.groupby('SK_ID_CURR')['DAYS_ENDDATE_FACT'].median() # most recently ended credit for a client

    tmp = bureau.loc[bureau.CREDIT_ACTIVE == 0, ['SK_ID_CURR', 'DAYS_CREDIT']]
    tmp = tmp.groupby('SK_ID_CURR')['DAYS_CREDIT'].median() # most recently applied credit at Home credit

    res   = tmp.div(res, fill_value=0)
    data.loc[:, 'median_ratio_end_curr_credit'] = data.SK_ID_CURR.map(res)
    
    # total credit sum by credit types
    res = bureau.groupby(['SK_ID_CURR', 'CREDIT_ACTIVE'])['AMT_CREDIT_SUM'].sum().unstack()
    res = res[0].divide(res[2], fill_value=np.nan)
    data.loc[:, 'total_credit_active_to_closed']  = data.SK_ID_CURR.map(res)
    
    # total credit sum debt by credit type
    res = bureau.groupby(['SK_ID_CURR', 'CREDIT_ACTIVE'])['AMT_CREDIT_SUM_DEBT'].sum().unstack()
    res = res[0].divide(res[2], fill_value=np.nan)
    data.loc[:, 'total_credit_debt_active_to_closed']  = data.SK_ID_CURR.map(res)

    # relationship between days credit update and total credit and debt amount reported
    # to the credit bureau
    days_credit_update_years = -bureau.DAYS_CREDIT_UPDATE / 365
    tmp = bureau.AMT_CREDIT_SUM.map(np.log1p) / days_credit_update_years
    tmp = tmp.groupby(bureau.SK_ID_CURR).sum()
    data.loc[:, 'sum_credit_latest_update'] = data.SK_ID_CURR.map(tmp)

    tmp = bureau.AMT_CREDIT_SUM_DEBT.map(np.log1p) / days_credit_update_years
    tmp = tmp.groupby(bureau.SK_ID_CURR).sum()
    data.loc[:, 'sum_debt_latest_update'] = data.SK_ID_CURR.map(tmp)

    del tmp, days_credit_update_years
    gc.collect()

    x = bureau.loc[bureau.CREDIT_ACTIVE == 2, ['SK_ID_CURR', 'AMT_CREDIT_SUM', 'AMT_CREDIT_MAX_OVERDUE']]
    x.loc[:, 'r'] = x.AMT_CREDIT_MAX_OVERDUE / x.AMT_CREDIT_SUM

    res = x.groupby('SK_ID_CURR')['r'].sum()
    data.loc[:, 'max_overdue_credit_sum'] = data.SK_ID_CURR.map(res)
    
    res = x.groupby('SK_ID_CURR')['r'].mean()
    data.loc[:, 'max_overdue_credit_mean'] = data.SK_ID_CURR.map(res)
    
    res = x.groupby('SK_ID_CURR')['r'].median()
    data.loc[:, 'max_overdue_credit_median'] = data.SK_ID_CURR.map(res)
    
    res = x.groupby('SK_ID_CURR')['r'].max()
    data.loc[:, 'max_overdue_credit_max'] = data.SK_ID_CURR.map(res)
    
    res = x.groupby('SK_ID_CURR')['r'].min()
    data.loc[:, 'max_overdue_credit_min'] = data.SK_ID_CURR.map(res)
    
    del x, res
    gc.collect()

    # difference between first bureau credit and employment date
    latest_credit_date = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].min()
    latest_credit_date = data.SK_ID_CURR.map(latest_credit_date)
    data.loc[:, 'diff_latest_credit_employed'] = latest_credit_date - data.DAYS_EMPLOYED.replace({365243: np.nan})

    del latest_credit_date
    gc.collect()

    # debt to number of days remaining for active credit reported to Credit Bureau
    debt      = bureau.loc[bureau.DAYS_CREDIT_ENDDATE > 0, 'AMT_CREDIT_SUM_DEBT']
    days_left = bureau.loc[bureau.DAYS_CREDIT_ENDDATE > 0, 'DAYS_CREDIT_ENDDATE']

    tmp       = debt / days_left
    tmp       = tmp.groupby(bureau.loc[bureau.DAYS_CREDIT_ENDDATE > 0, 'SK_ID_CURR']).mean()
    data.loc[:, 'debt_times_days_left_bureau_mean'] = data.SK_ID_CURR.map(tmp)

    tmp       = debt * days_left
    tmp       = tmp.groupby(bureau.loc[bureau.DAYS_CREDIT_ENDDATE > 0, 'SK_ID_CURR']).mean()
    data.loc[:, 'debt_to_days_left_bureau_mean'] = data.SK_ID_CURR.map(tmp)

    # credit to debt difference by remaining days
    mask = (bureau.CREDIT_ACTIVE == 0) & (bureau.AMT_CREDIT_SUM_DEBT > 0) & (bureau.DAYS_CREDIT_ENDDATE > 0)
    diff_credit_debt = bureau.loc[mask, 'AMT_CREDIT_SUM'] - bureau.loc[mask, 'AMT_CREDIT_SUM_DEBT']
    ratio_diff_left_days = (diff_credit_debt / bureau.loc[mask, 'DAYS_CREDIT_ENDDATE']).replace([-np.inf, np.inf], np.nan)
    
    tmp = ratio_diff_left_days.groupby(bureau.loc[mask, 'SK_ID_CURR']).mean()
    data.loc[:, 'mean_ratio_diff_credit_debt_left_days'] = data.SK_ID_CURR.map(tmp)

    tmp = ratio_diff_left_days.groupby(bureau.loc[mask, 'SK_ID_CURR']).min()
    data.loc[:, 'min_ratio_diff_credit_debt_left_days'] = data.SK_ID_CURR.map(tmp)
    
    tmp = ratio_diff_left_days.groupby(bureau.loc[mask, 'SK_ID_CURR']).max()
    data.loc[:, 'max_ratio_diff_credit_debt_left_days'] = data.SK_ID_CURR.map(tmp)

    tmp = ratio_diff_left_days.groupby(bureau.loc[mask, 'SK_ID_CURR']).sum()
    data.loc[:, 'sum_ratio_diff_credit_debt_left_days'] = data.SK_ID_CURR.map(tmp)

    # credit to debt difference by remaning days and age, employed since

    mask = (bureau.CREDIT_ACTIVE == 0) & (bureau.AMT_CREDIT_SUM_DEBT > 0) & (bureau.DAYS_CREDIT_ENDDATE > 0)
    diff_credit_debt = bureau.loc[mask, 'AMT_CREDIT_SUM'] - bureau.loc[mask, 'AMT_CREDIT_SUM_DEBT']
    ratio_diff_left_days = (diff_credit_debt / bureau.loc[mask, 'DAYS_CREDIT_ENDDATE']).replace([-np.inf, np.inf], np.nan)
    
    tmp = ratio_diff_left_days.groupby(bureau.loc[mask, 'SK_ID_CURR']).mean()
    data.loc[:, 'mean_ratio_diff_credit_debt_left_days_age_employed'] = (data.SK_ID_CURR.map(tmp)) * ((-data.DAYS_EMPLOYED.replace({365243: np.nan}) / 365) / (-data.DAYS_BIRTH / 365))

    tmp = ratio_diff_left_days.groupby(bureau.loc[mask, 'SK_ID_CURR']).min()
    data.loc[:, 'min_ratio_diff_credit_debt_left_days_age_employed'] = (data.SK_ID_CURR.map(tmp)) * ((-data.DAYS_EMPLOYED.replace({365243: np.nan}) / 365) / (-data.DAYS_BIRTH / 365))
    
    tmp = ratio_diff_left_days.groupby(bureau.loc[mask, 'SK_ID_CURR']).max()
    data.loc[:, 'max_ratio_diff_credit_debt_left_days_age_employed'] = (data.SK_ID_CURR.map(tmp)) * ((-data.DAYS_EMPLOYED.replace({365243: np.nan}) / 365) / (-data.DAYS_BIRTH / 365))

    tmp = ratio_diff_left_days.groupby(bureau.loc[mask, 'SK_ID_CURR']).sum()
    data.loc[:, 'sum_ratio_diff_credit_debt_left_days_age_employed'] = (data.SK_ID_CURR.map(tmp)) * ((-data.DAYS_EMPLOYED.replace({365243: np.nan}) / 365) / (-data.DAYS_BIRTH / 365))

    # relationship between total debt from bureau credits with age
    mask = (bureau.CREDIT_ACTIVE == 0) & (bureau.DAYS_CREDIT_ENDDATE > 0)
    total_debt_bureau   = bureau.loc[mask, :].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum()
    total_credit_bureau = bureau.loc[mask, :].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum()

    total_debt_bureau   = data.SK_ID_CURR.map(total_debt_bureau)
    total_credit_bureau = data.SK_ID_CURR.map(total_credit_bureau) 
    
    data.loc[:, 'debt_credit_bureau_ratio_with_age'] = (total_debt_bureau / total_credit_bureau) * (-data.DAYS_BIRTH / 365)

    del total_debt_bureau, total_credit_bureau
    gc.collect()

    # relationship between bureau credit end date and employed since
    mask = (bureau.CREDIT_ACTIVE == 0) & (bureau.DAYS_CREDIT_ENDDATE > 0)
    total_enddate = bureau.loc[mask, :].groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].sum()
    total_enddate = data.SK_ID_CURR.map(total_enddate)
    
    mean_enddate = bureau.loc[mask, :].groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].mean()
    mean_enddate = data.SK_ID_CURR.map(mean_enddate)

    max_enddate = bureau.loc[mask, :].groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].max()
    max_enddate = data.SK_ID_CURR.map(max_enddate)

    min_enddate = bureau.loc[mask, :].groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].min()
    min_enddate = data.SK_ID_CURR.map(min_enddate)

    data.loc[:, 'total_bureau_enddate_employed_since'] = total_enddate / (-data.DAYS_EMPLOYED.replace({365243: np.nan}) / 365)
    data.loc[:, 'mean_bureau_enddate_employed_since'] = mean_enddate / (-data.DAYS_EMPLOYED.replace({365243: np.nan}) / 365)
    data.loc[:, 'max_bureau_enddate_employed_since'] = max_enddate / (-data.DAYS_EMPLOYED.replace({365243: np.nan}) / 365)
    data.loc[:, 'min_bureau_enddate_employed_since'] = min_enddate / (-data.DAYS_EMPLOYED.replace({365243: np.nan}) / 365)
    
    # debt to credit ratio with external scores
    mask = (bureau.CREDIT_ACTIVE == 0) & (bureau.DAYS_CREDIT_ENDDATE > 0)
    total_credit = bureau.loc[mask, :].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum()
    max_credit   = bureau.loc[mask, :].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].max()

    total_credit = data.SK_ID_CURR.map(total_credit)
    max_credit   = data.SK_ID_CURR.map(max_credit)

    total_debt = bureau.loc[mask, :].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum()
    max_debt   = bureau.loc[mask, :].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].max()

    total_debt = data.SK_ID_CURR.map(total_debt)
    max_debt   = data.SK_ID_CURR.map(max_debt)

    data.loc[:, 'debt_to_credit_ext_source_2_mult']     = (total_debt / total_credit) * data.EXT_SOURCE_2
    data.loc[:, 'max_debt_to_credit_ext_source_2_mult'] = (max_debt / max_credit) * data.EXT_SOURCE_2

    return data, list(set(data.columns) - set(COLS))

#TODO: Features here should be moved to feature engineering in corresponding model file.
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
    res = bureau.loc[:, ['SK_ID_CURR', 'SK_ID_BUREAU']].merge(res, how='left').drop('SK_ID_BUREAU', axis=1)
    res = res.groupby('SK_ID_CURR')[0].sum()
    data.loc[:, 'mean_status'] = data.SK_ID_CURR.map(res)

    del res
    gc.collect()

    res = bureau_bal.groupby('SK_ID_BUREAU').size().reset_index().rename(columns={0: 'num_count'})
    res = bureau.merge(res, how='left')
    
    median_num_bureau_balance = res.groupby('SK_ID_CURR')['num_count'].median()
    data.loc[:, 'median_num_bureau_balance'] = data.SK_ID_CURR.map(median_num_bureau_balance)

    mean_num_bureau_balance = res.groupby('SK_ID_CURR')['num_count'].mean()
    data.loc[:, 'mean_num_bureau_balance'] = data.SK_ID_CURR.map(mean_num_bureau_balance)

    max_num_bureau_balance = res.groupby('SK_ID_CURR')['num_count'].max()
    data.loc[:, 'max_num_bureau_balance'] = data.SK_ID_CURR.map(max_num_bureau_balance)

    min_num_bureau_balance = res.groupby('SK_ID_CURR')['num_count'].min()
    data.loc[:, 'min_num_bureau_balance'] = data.SK_ID_CURR.map(min_num_bureau_balance)

    std_num_bureau_balance = res.groupby('SK_ID_CURR')['num_count'].min()
    data.loc[:, 'std_num_bureau_balance'] = data.SK_ID_CURR.map(std_num_bureau_balance)

    del res
    gc.collect()

    # treat status as string of characters
    # this takes up to 4 minutes
    first_character = bureau_bal.groupby(['SK_ID_BUREAU'])['STATUS'].first()
    res = bureau.SK_ID_BUREAU.map(first_character)
    tmp = bureau.groupby(['SK_ID_CURR', res]).size().unstack().fillna(0).astype(np.int).reset_index()
    tmp.columns = [f'STATUS{col}' if col != 'SK_ID_CURR' else col for col in tmp.columns]
    
    data        = merge(data, tmp)

    del tmp, res, first_character
    gc.collect()

    tmp = bureau.loc[bureau.CREDIT_ACTIVE == 0, ['SK_ID_CURR', 'SK_ID_BUREAU']]\
                .merge(bureau_bal, on='SK_ID_BUREAU', how='left')
    
    tmp.loc[:, 'status_unk'] = tmp.STATUS == 6

    status_unk  = tmp.groupby(['SK_ID_CURR', 'SK_ID_BUREAU'])['status_unk'].sum()
    status_size = tmp.groupby(['SK_ID_CURR', 'SK_ID_BUREAU']).size()

    res = status_unk / status_size

    res = res.reset_index().drop('SK_ID_BUREAU', axis=1)
    res = res.groupby('SK_ID_CURR')[0].mean()
    data.loc[:, 'completed_to_total'] = data.SK_ID_CURR.map(res)

    del tmp, res, status_size, status_unk
    gc.collect()
    
    # Relationship between count of different status ( most recent ) for credits
    # reported by Bureau and TARGET

    data_bureau = data.loc[:, ['SK_ID_CURR', 'TARGET']]\
                           .merge(bureau.loc[:, ['SK_ID_CURR',
                                  'SK_ID_BUREAU'
                                 ]], on='SK_ID_CURR', how='left')

    most_recent_status = bureau_bal.groupby('SK_ID_BUREAU', as_index=False)['MONTHS_BALANCE'].max()
    bureau_bal_recent  = bureau_bal.merge(most_recent_status, on=['SK_ID_BUREAU', 'MONTHS_BALANCE'], how='inner')

    del most_recent_status
    gc.collect()

    data_bureau_bal = data_bureau.merge(bureau_bal_recent, on='SK_ID_BUREAU', how='left')

    data_bureau_bal.loc[:, 'STATUS']  = data_bureau_bal.STATUS.astype(np.str)
    data_bureau_bal.loc[:, 'STATUS']  = data_bureau_bal.STATUS.fillna('missing')

    ss = data_bureau_bal.groupby(['SK_ID_CURR', 'STATUS']).size().unstack().fillna(0)
    print(ss.columns)
    ss.loc[:, 'ratio_C_X'] = (ss['6.0'] + ss['7.0']) / ss['nan']

    data.loc[:, 'ratio_sum_C_X_to_missing'] = data.SK_ID_CURR.map(ss.ratio_C_X)
    
    del ss, bureau_bal_recent, data_bureau_bal
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

    # ratio of annuity to credit, cnt_payment ( median )
    res = ((prev_app.AMT_ANNUITY / prev_app.AMT_CREDIT) * prev_app.CNT_PAYMENT).replace([np.inf, -np.inf], np.nan)
    res = res.groupby(prev_app.SK_ID_CURR).median()
    data.loc[:, 'past_annuity_to_credit_cnt_payment_median'] = data.SK_ID_CURR.map(res).astype(np.float32)
    
    # ratio of annuity to credit, cnt_payment ( median )
    res = ((prev_app.AMT_ANNUITY / prev_app.AMT_CREDIT) * prev_app.CNT_PAYMENT).replace([np.inf, -np.inf], np.nan)
    res = res.groupby(prev_app.SK_ID_CURR).mean()
    data.loc[:, 'past_annuity_to_credit_cnt_payment_mean'] = data.SK_ID_CURR.map(res).astype(np.float32)
    
    # ratio of annuity to credit, cnt_payment ( max )
    res = ((prev_app.AMT_ANNUITY / prev_app.AMT_CREDIT) * prev_app.CNT_PAYMENT).replace([np.inf, -np.inf], np.nan)
    res = res.groupby(prev_app.SK_ID_CURR).max()
    data.loc[:, 'past_annuity_to_credit_cnt_payment_max'] = data.SK_ID_CURR.map(res).astype(np.float32)
    
    # ratio of annuity to credit, cnt_payment ( std )
    res = ((prev_app.AMT_ANNUITY / prev_app.AMT_CREDIT) * prev_app.CNT_PAYMENT).replace([np.inf, -np.inf], np.nan)
    res = res.groupby(prev_app.SK_ID_CURR).std()
    data.loc[:, 'past_annuity_to_credit_cnt_payment_std'] = data.SK_ID_CURR.map(res).astype(np.float32)
    
    del res
    gc.collect()

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

    # mean of term of previous applications
    mean_prev_app = prev_app.groupby('SK_ID_CURR')['CNT_PAYMENT'].mean()
    data.loc[:, 'mean_prev_app'] = data.SK_ID_CURR.map(mean_prev_app).astype(np.float32)

    # std of previous applications
    std_prev_app  = prev_app.groupby('SK_ID_CURR')['CNT_PAYMENT'].std()
    data.loc[:, 'std_prev_app']  = data.SK_ID_CURR.map(std_prev_app)

    # skew of previous applications
    skew_prev_app = prev_app.groupby('SK_ID_CURR')['CNT_PAYMENT'].skew()
    data.loc[:, 'skew_prev_app'] = data.SK_ID_CURR.map(skew_prev_app)

    del mean_prev_app, std_prev_app, skew_prev_app
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

    # difference between termination of credit and day decision was made ( aggregate statistics )
    mask = prev_app.NAME_CONTRACT_STATUS == 0
    diff_termination_decision                = prev_app.loc[mask].DAYS_TERMINATION.replace({365243: np.nan}) - prev_app.loc[mask].DAYS_DECISION
    diff_termination_decision                = diff_termination_decision.groupby(prev_app.SK_ID_CURR).mean()
    data.loc[:, 'diff_termination_decision'] = data.SK_ID_CURR.map(diff_termination_decision).astype(np.float32)
    
    diff_termination_decision                    = prev_app.loc[mask].DAYS_TERMINATION.replace({365243: np.nan}) - prev_app.loc[mask].DAYS_DECISION    
    diff_termination_decision_sum                = diff_termination_decision.groupby(prev_app.SK_ID_CURR).sum()    
    data.loc[:, 'diff_termination_decision_sum'] = data.SK_ID_CURR.map(diff_termination_decision_sum)

    diff_termination_decision                    = prev_app.loc[mask].DAYS_TERMINATION.replace({365243: np.nan}) - prev_app.loc[mask].DAYS_DECISION    
    diff_termination_decision_min                = diff_termination_decision.groupby(prev_app.SK_ID_CURR).min()    
    data.loc[:, 'diff_termination_decision_min'] = data.SK_ID_CURR.map(diff_termination_decision_min)


    del diff_termination_decision, diff_termination_decision_sum, diff_termination_decision_min
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


    res  = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR',
                        'AMT_ANNUITY',
                        'AMT_CREDIT',
                        'CNT_PAYMENT']]

    tmp  = res.AMT_CREDIT / (res.AMT_ANNUITY * res.CNT_PAYMENT) 
    tmp  = tmp.groupby(res.SK_ID_CURR).mean()
    data.loc[:, 'prev_app_rate']  = data.SK_ID_CURR.map(tmp)

    del res, tmp
    gc.collect()

    max_credit_term = prev_app[prev_app.NAME_CONTRACT_STATUS == 0].groupby('SK_ID_CURR')['CNT_PAYMENT'].max()
    data.loc[:, 'max_credit_term'] = data.SK_ID_CURR.map(max_credit_term)
    
    min_credit_term = prev_app[prev_app.NAME_CONTRACT_STATUS == 0].groupby('SK_ID_CURR')['CNT_PAYMENT'].max()
    data.loc[:, 'min_credit_term'] = data.SK_ID_CURR.map(min_credit_term)
    
    data.loc[:, 'diff_max_min_credit_term'] = data.max_credit_term.subtract(data.min_credit_term, fill_value=0)
    
    
    res                                         = prev_app[prev_app.NAME_CONTRACT_STATUS == 0].groupby('SK_ID_CURR')['DAYS_DECISION'].min()
    data.loc[:, 'most_oldest_prev_application'] = data.SK_ID_CURR.map(res)

    res = prev_app.groupby('SK_ID_CURR')['DAYS_DECISION'].min()
    data.loc[:, 'new_user_date'] = data.SK_ID_CURR.map(res)

    
    res                        = prev_app[prev_app.NAME_CONTRACT_STATUS == 0].groupby('SK_ID_CURR')['DAYS_DECISION'].median()
    data.loc[:, 'median_prev_application']  = data.SK_ID_CURR.map(res)
    
    del res, max_credit_term, min_credit_term
    gc.collect()


    d = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, 'DAYS_DECISION']
    f = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, 'DAYS_FIRST_DUE'].replace({365243: np.nan})
    t = f.subtract(d)

    t = t.groupby(prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, 'SK_ID_CURR']).std()
    data.loc[:, 'diff_first_due_days_var'] = data.SK_ID_CURR.map(t)

    del d, f, t
    gc.collect()

    tmp = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0) &\
                   (prev_app.CNT_PAYMENT > 0)
                   , ['SK_ID_CURR',
                      'AMT_CREDIT',
                      'AMT_ANNUITY',
                      'CNT_PAYMENT',
                      'DAYS_DECISION'
                     ]]

    tmp.loc[:, 'per_month_annuity'] = tmp.AMT_CREDIT / tmp.CNT_PAYMENT
    tmp.loc[:, 'diff_annuity']      = tmp.AMT_ANNUITY - tmp.per_month_annuity
    tmp.loc[:, 'ratio_annuity']     = tmp.AMT_ANNUITY / tmp.per_month_annuity
    tmp.loc[:, 'mult_ratio_annuity_days_decision'] = tmp.ratio_annuity * (-tmp.DAYS_DECISION / 365)


    res = tmp.groupby('SK_ID_CURR')['per_month_annuity'].median()
    data.loc[:, 'prev_app_ratio_annuity'] = data.SK_ID_CURR.map(res)

    res = tmp.groupby('SK_ID_CURR')['mult_ratio_annuity_days_decision'].mean()
    data.loc[:, 'mult_ratio_annuity_days_decision_mean'] = data.SK_ID_CURR.map(res)

    res = tmp.groupby('SK_ID_CURR')['mult_ratio_annuity_days_decision'].max()
    data.loc[:, 'mult_ratio_annuity_days_decision_max'] = data.SK_ID_CURR.map(res)
    
    del tmp, res
    gc.collect()

    # difference between first previous application approved credit and employment date
    first_prev_app_credit = prev_app[prev_app.NAME_CONTRACT_STATUS  == 0].groupby('SK_ID_CURR')['DAYS_DECISION'].min()
    first_prev_app_credit = data.SK_ID_CURR.map(first_prev_app_credit)
    data.loc[:, 'diff_first_prev_app_credit_employed'] = first_prev_app_credit - data.DAYS_EMPLOYED.replace({365243: np.nan})

    # Represent contract status of previous application as string of characters
    contract_status_str = prev_app.sort_values(by='DAYS_DECISION', ascending=False).groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS']\
                                  .apply(lambda x: ''.join([str(z) for z in x]))
    
    contract_status_str        = data.SK_ID_CURR.map(contract_status_str)
    contract_status_str_counts = contract_status_str.value_counts()

    contract_status_str[contract_status_str.isin((contract_status_str_counts <= 8).index.values)] = '-1'
    contract_status_str = pd.factorize(contract_status_str)[0]
    data.loc[:, 'contract_status_str'] = contract_status_str

    del contract_status_str
    gc.collect()

    # max annuity of the approved previous credit and current annuity
    max_annuity_prev_app = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR', 'AMT_ANNUITY']]\
                               .groupby('SK_ID_CURR')['AMT_ANNUITY'].max()

    max_annuity_prev_app = data.SK_ID_CURR.map(max_annuity_prev_app)
    data.loc[:, 'ratio_max_annuity_prev_app_curr_annuity'] = (data.AMT_ANNUITY / max_annuity_prev_app).replace([-np.inf, np.inf], np.nan)

    # months still left on payment
    mask = (prev_app.NAME_CONTRACT_STATUS == 0) & (prev_app.CNT_PAYMENT > 0) &\
           ((-prev_app.DAYS_DECISION / 30) < (prev_app.CNT_PAYMENT))

    months_left_to_pay = (prev_app.loc[mask, 'CNT_PAYMENT']) - (-prev_app.loc[mask, 'DAYS_DECISION'] / 30) 
    months_left_to_pay = months_left_to_pay.groupby(prev_app.loc[mask, 'SK_ID_CURR']).mean()

    data.loc[:, 'months_left_to_pay'] = data.SK_ID_CURR.map(months_left_to_pay)

    # difference between actual and proposed termination
    mask = (prev_app.DAYS_TERMINATION.notnull()) & (prev_app.DAYS_TERMINATION != 365243)

    a = prev_app.loc[mask,'CNT_PAYMENT'] - (-prev_app.loc[mask, 'DAYS_DECISION'] / 30)
    b = prev_app.loc[mask, 'DAYS_TERMINATION']

    tmp = (a - b).groupby(prev_app.loc[mask, 'SK_ID_CURR']).mean()
    data.loc[:, 'actual_proposed_termination'] = data.SK_ID_CURR.map(tmp)

    del a, b, tmp
    gc.collect()

    # relationship between number of loans in last 6 months compared
    # with number of loans in 12 months
    last_6_months = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0) &\
                             (prev_app.DAYS_DECISION >= -180)
                            ]

    last_6_months = last_6_months.groupby('SK_ID_CURR').size()
    last_6_months = data.SK_ID_CURR.map(last_6_months)

    last_12_months = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0) &\
                                (prev_app.DAYS_DECISION < -180) &\
                                (prev_app.DAYS_DECISION >= -365)
                                ]

    last_12_months = last_12_months.groupby('SK_ID_CURR').size()
    last_12_months = data.SK_ID_CURR.map(last_12_months)

    data.loc[:, 'ratio_6_12_prev_app_months'] = last_6_months.divide(last_12_months, fill_value=np.nan)
    data.loc[:, 'diff_6_12_prev_app_months']  = last_12_months.subtract(last_6_months, fill_value=0)

    del last_6_months, last_12_months
    gc.collect()

    # most recent and oldest dpd month in pos cash history
    dpd_instances = pos_cash.loc[pos_cash.SK_DPD > 0, :]
    
    oldest_dpd    = dpd_instances.groupby('SK_ID_CURR')['MONTHS_BALANCE'].min()
    latest_dpd    = dpd_instances.groupby('SK_ID_CURR')['MONTHS_BALANCE'].max()

    data.loc[:, 'oldest_dpd_month_pos_cash'] = data.SK_ID_CURR.map(oldest_dpd)
    data.loc[:, 'latest_dpd_month_pos_cash'] = data.SK_ID_CURR.map(latest_dpd)
    
    del dpd_instances, oldest_dpd, latest_dpd
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


    # installments left on cash and consumer loans for a user.
    tmp = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['MONTHS_BALANCE'].max()
    tmp = tmp.merge(pos_cash, how='inner')
    
    t = tmp.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].median()
    data.loc[:, 'median_pos_installments_left'] = data.SK_ID_CURR.map(t)

    t = tmp.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].min()
    data.loc[:, 'min_pos_installments_left'] = data.SK_ID_CURR.map(t)

    t = tmp.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].mean()
    data.loc[:, 'mean_pos_installments_left'] = data.SK_ID_CURR.map(t) 

    t = tmp.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].var()
    data.loc[:, 'var_pos_installments_left'] = data.SK_ID_CURR.map(t) 
    
    total_dpd = tmp.groupby('SK_ID_CURR')['SK_DPD'].sum()
    data.loc[:, 'most_recent_total_pos_cash_dpd'] = data.SK_ID_CURR.map(total_dpd)

    mean_dpd  = tmp.groupby('SK_ID_CURR')['SK_DPD'].mean()
    data.loc[:, 'most_recent_mean_pos_cash_dpd'] = data.SK_ID_CURR.map(mean_dpd)

    max_dpd  = tmp.groupby('SK_ID_CURR')['SK_DPD'].max()
    data.loc[:, 'most_recent_max_pos_cash_dpd'] = data.SK_ID_CURR.map(max_dpd)

    min_dpd  = tmp.groupby('SK_ID_CURR')['SK_DPD'].min()
    data.loc[:, 'most_recent_min_pos_cash_dpd'] = data.SK_ID_CURR.map(min_dpd)

    std_dpd  = tmp.groupby('SK_ID_CURR')['SK_DPD'].std()
    data.loc[:, 'most_recent_std_pos_cash_dpd'] = data.SK_ID_CURR.map(std_dpd)

    total_credits = pos_cash.groupby('SK_ID_CURR').size()
    tmp.loc[:, 'recent_status'] = (tmp.NAME_CONTRACT_STATUS == 4).astype(np.uint8)
    completed_credits        = tmp.groupby('SK_ID_CURR')['recent_status'].sum()
    ratio_completed_to_total = (completed_credits / total_credits).fillna(0)

    data.loc[:, 'ratio_completed_to_total_pos_cash'] = data.SK_ID_CURR.map(ratio_completed_to_total)

    return data, list(set(data.columns) - set(COLS))

def credit_card_features(credit_bal, data):
    COLS = data.columns.tolist()
    
    # ratio of balance to total credit limit actual

    total_credit_limit = credit_bal.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].sum()
    total_balance      = credit_bal.groupby('SK_ID_CURR')['AMT_BALANCE'].sum() 
    tmp                = total_balance / total_credit_limit
    data.loc[:, 'ratio_credit_bal_limit'] = data.SK_ID_CURR.map(tmp)

    del tmp
    gc.collect()

    # difference between credit limit and balance
    tmp = total_credit_limit - total_balance
    data.loc[:, 'diff_credit_bal_limit'] = data.SK_ID_CURR.map(tmp)

    del tmp
    gc.collect()

    # total balance
    data.loc[:, 'credit_total_balance'] = data.SK_ID_CURR.map(total_balance)

    # total credit limit
    data.loc[:, 'credit_total_limit']   = data.SK_ID_CURR.map(total_credit_limit)

    # total defaults
    total_defaults = credit_bal.groupby('SK_ID_CURR')['SK_DPD'].sum()
    data.loc[:, 'credit_total_dpds'] = data.SK_ID_CURR.map(total_defaults)

    # total regular installments
    instalment_regular = credit_bal.groupby('SK_ID_CURR')['AMT_INST_MIN_REGULARITY'].sum()
    data.loc[:, 'credit_total_instalment_regular'] = data.SK_ID_CURR.map(instalment_regular)

    # total payment current
    payment_current = credit_bal.groupby('SK_ID_CURR')['AMT_PAYMENT_CURRENT'].sum()
    data.loc[:, 'credit_total_payment_current'] = data.SK_ID_CURR.map(instalment_regular)

    # mean of amount balance during previous payments
    mean_amt_balance                = credit_bal.groupby('SK_ID_CURR')['AMT_BALANCE'].mean()
    data.loc[:, 'mean_amt_balance'] = data.SK_ID_CURR.map(mean_amt_balance)

    # mean of actual credit limit
    mean_credit_limit                = credit_bal.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].mean()
    data.loc[:, 'mean_credit_limit'] = data.SK_ID_CURR.map(mean_credit_limit).astype(np.float32)

    # total paid installments on previous credit
    total_paid_installments                = credit_bal.groupby('SK_ID_CURR')['CNT_INSTALMENT_MATURE_CUM'].sum()
    data.loc[:, 'total_paid_installments'] = data.SK_ID_CURR.map(total_paid_installments)

    # maximum number of credit card installments remaining
    max_total_installments                 = credit_bal.groupby('SK_ID_CURR')['CNT_INSTALMENT_MATURE_CUM'].max()
    data.loc[:, 'max_total_installments']  = data.SK_ID_CURR.map(max_total_installments)

    del max_total_installments
    gc.collect()

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
    diff_min_installment_balance                = diff_min_installment_balance.groupby(credit_bal.SK_ID_CURR).mean()
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

    # total number of drawings
    res = credit_bal.loc[credit_bal.MONTHS_BALANCE > -9, :]
    res = res.groupby('SK_ID_CURR').CNT_DRAWINGS_CURRENT.sum()
    data.loc[:, 'total_atm_transactions'] = data.SK_ID_CURR.map(res)

    del res
    gc.collect()

    del mean_amt_balance, mean_credit_limit
    del total_paid_installments, mean_total_drawings, diff_bal_credit
    del ratio_min_installment_balance, diff_min_installment_balance
    gc.collect()

    # credit card line usage
    tmp = credit_bal.loc[(credit_bal.NAME_CONTRACT_STATUS == 0) &\
                     (credit_bal.AMT_CREDIT_LIMIT_ACTUAL > 0)
                     , ['SK_ID_CURR', 'AMT_BALANCE',
                        'AMT_CREDIT_LIMIT_ACTUAL'
                     ]]

    t    = tmp.AMT_BALANCE / tmp.AMT_CREDIT_LIMIT_ACTUAL
    t1   = t.groupby(tmp.SK_ID_CURR).sum()
    t2   = t.groupby(tmp.SK_ID_CURR).mean()
    
    data.loc[:, 'sum_line_usage']  = data.SK_ID_CURR.map(t1)
    data.loc[:, 'mean_line_usage'] = data.SK_ID_CURR.map(t2)

    del tmp, t1, t2
    gc.collect()

    t   = credit_bal[credit_bal.NAME_CONTRACT_STATUS == 0].CNT_INSTALMENT_MATURE_CUM == 0
    t1  = t.groupby(credit_bal[credit_bal.NAME_CONTRACT_STATUS == 0].SK_ID_CURR).sum()
    r   = t.groupby(credit_bal[credit_bal.NAME_CONTRACT_STATUS == 0].SK_ID_CURR).size()
    tmp = t1 / r

    data.loc[:, 'ratio_zero_installments'] = data.SK_ID_CURR.map(tmp)
    
    del t, t1, r, tmp
    gc.collect()

    # total credit card load
    total_credits                     = credit_bal.groupby('SK_ID_CURR').size()
    total_insallments_paid            = credit_bal.groupby('SK_ID_CURR')['CNT_INSTALMENT_MATURE_CUM'].sum()
    total_installments_across_credits = total_credits.multiply(total_insallments_paid, fill_value=1)

    data['total_installments_across_credits'] = data.SK_ID_CURR.map(total_installments_across_credits)
    
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

    tmp = (installments.DAYS_ENTRY_PAYMENT - installments.DAYS_INSTALMENT)
    tmp = tmp.groupby(installments.SK_ID_CURR).median()

    data.loc[:, 'median_diff_actual_decided'] = data.SK_ID_CURR.map(tmp)

    tmp = (installments.DAYS_ENTRY_PAYMENT - installments.DAYS_INSTALMENT)
    tmp = tmp.groupby(installments.SK_ID_CURR).min()

    data.loc[:, 'min_diff_actual_decided'] = data.SK_ID_CURR.map(tmp)

    tmp = (installments.DAYS_ENTRY_PAYMENT - installments.DAYS_INSTALMENT)
    tmp = tmp.groupby(installments.SK_ID_CURR).max()

    data.loc[:, 'max_diff_actual_decided'] = data.SK_ID_CURR.map(tmp)

    tmp = (installments.DAYS_ENTRY_PAYMENT - installments.DAYS_INSTALMENT)
    tmp = tmp.groupby(installments.SK_ID_CURR).mean()

    data.loc[:, 'mean_diff_actual_decided'] = data.SK_ID_CURR.map(tmp)

    # difference between installment amount and paid amount for late days
    td  = installments.DAYS_ENTRY_PAYMENT - installments.DAYS_INSTALMENT
    tad = installments.AMT_PAYMENT - installments.AMT_INSTALMENT

    x   = pd.DataFrame({'td': td, 'tad': tad})
    
    # mean
    tmp = x[x['td'] > 0].groupby(installments.SK_ID_CURR)['tad'].apply(np.mean)
    data.loc[:, 'mean_diff_installment_actual_late_days'] = data.SK_ID_CURR.map(tmp)
    
    # median
    tmp = x[x['td'] > 0].groupby(installments.SK_ID_CURR)['tad'].apply(np.median)
    data.loc[:, 'median_diff_installment_actual_late_days'] = data.SK_ID_CURR.map(tmp)
    
    # max
    tmp = x[x['td'] > 0].groupby(installments.SK_ID_CURR)['tad'].apply(np.max)
    data.loc[:, 'max_diff_installment_actual_late_days'] = data.SK_ID_CURR.map(tmp)
    
    # min
    tmp = x[x['td'] > 0].groupby(installments.SK_ID_CURR)['tad'].apply(np.min)
    data.loc[:, 'min_diff_installment_actual_late_days'] = data.SK_ID_CURR.map(tmp)
    
    # sum
    tmp = x[x['td'] > 0].groupby(installments.SK_ID_CURR)['tad'].apply(np.sum)
    data.loc[:, 'sum_diff_installment_actual_late_days'] = data.SK_ID_CURR.map(tmp)


    # ratio of installment to be paid versus actual amount paid

    res = (installments.AMT_INSTALMENT / installments.AMT_PAYMENT).replace([np.inf, -np.inf], np.nan)
    res = res.groupby(installments.SK_ID_CURR).mean()

    data.loc[:, 'ratio_actual_decided_amount'] = data.SK_ID_CURR.map(res)

    del mean_installment, res
    gc.collect()

    # differenece between actual installment amount vs what was paid

    res = (installments.AMT_INSTALMENT - installments.AMT_PAYMENT)
    res = res.groupby(installments.SK_ID_CURR).mean()

    data.loc[:, 'mean_diff_actual_decided_amount'] = data.SK_ID_CURR.map(res)

    res = (installments.AMT_INSTALMENT - installments.AMT_PAYMENT)
    res = res.groupby(installments.SK_ID_CURR).median()

    data.loc[:, 'median_diff_actual_decided_amount'] = data.SK_ID_CURR.map(res)


    del res
    gc.collect()

    data.loc[:, 'ratio_sum_payments_income'] = data.AMT_PAYMENT_sum.div(data.AMT_INCOME_TOTAL, fill_value=np.nan)

    # compare installment number in case of late payments
    
    tmp  = installments.DAYS_INSTALMENT - installments.DAYS_ENTRY_PAYMENT
    tmp  = installments.loc[tmp < 0, ['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']]

    tmp = installments.groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['NUM_INSTALMENT_NUMBER'].max()
    res = installments.groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['NUM_INSTALMENT_NUMBER'].min()

    tmp = tmp.merge(res, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')
    tmp.loc[:, 'd'] = tmp.NUM_INSTALMENT_NUMBER_x / tmp.NUM_INSTALMENT_NUMBER_y
    res = tmp.groupby('SK_ID_CURR')['d'].sum()

    data.loc[:, 'late_days_instalment_num_ratio'] = data.SK_ID_CURR.map(res)

    del tmp, res
    gc.collect()

    # number of late days in payment
    is_late = installments.DAYS_INSTALMENT < installments.DAYS_ENTRY_PAYMENT
    is_late = is_late.groupby(installments.SK_ID_CURR).sum()
    data.loc[:, 'num_late_payments_in_installments'] = data.SK_ID_CURR.map(is_late)

    del is_late
    gc.collect()

    # ratio of max and min installment during the course of prev home credits
    max_installment = installments.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['AMT_INSTALMENT'].max()
    min_installment = installments.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['AMT_INSTALMENT'].min()

    ratio_max_min   = max_installment.divide(min_installment, fill_value=np.nan).replace([-np.inf, np.inf], np.nan)

    tmp = ratio_max_min.reset_index()
    tmp = tmp.groupby('SK_ID_CURR')['AMT_INSTALMENT'].sum()

    data.loc[:, 'max_to_min_instalment_amount_prev_credit'] = data.SK_ID_CURR.map(tmp)

    del tmp
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

    del dcn, dd, dcu, df, res
    gc.collect()


    res = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR',
                                                           'DAYS_TERMINATION'
                                                          ]]\
        .merge(bureau.loc[bureau.CREDIT_ACTIVE == 2, ['SK_ID_CURR', 'DAYS_CREDIT']])
    res.loc[:, 'DAYS_TERMINATION'] = res.DAYS_TERMINATION.replace({365243: np.nan})
    
    res.loc[:, 'prev_app_end_bureau_start'] = res.DAYS_CREDIT - res.DAYS_TERMINATION
    tmp = res.groupby('SK_ID_CURR')['prev_app_end_bureau_start'].min()

    data.loc[:, 'prev_app_end_bureau_start_min'] = data.SK_ID_CURR.map(tmp)
    
    tmp = res.groupby('SK_ID_CURR')['prev_app_end_bureau_start'].max()
    data.loc[:, 'prev_app_end_bureau_start_max'] = data.SK_ID_CURR.map(tmp)
    
    del res, tmp
    gc.collect()

    # total active loans reported by Credit Bureau and those held at Home Credit
    bureau_loans       = bureau.loc[bureau.CREDIT_ACTIVE == 0].groupby('SK_ID_CURR').size()
    home_credit_loans  = prev_app.loc[(prev_app.DAYS_TERMINATION.isnull()) |\
                                     (prev_app.DAYS_TERMINATION == 365243.0), :].groupby('SK_ID_CURR').size()
    total_active_loans = bureau_loans.add(home_credit_loans, fill_value=0)
    data.loc[:, 'total_active_loans_across_credits'] = data.SK_ID_CURR.map(total_active_loans)


    del bureau_loans, home_credit_loans, total_active_loans
    gc.collect()

    # relationship between loan start date from bureau and previous home credit applications
    # and employment start date
    mask   = bureau.CREDIT_ACTIVE == 0
    bureau_credit_start = bureau.loc[mask, ['SK_ID_CURR', 'DAYS_CREDIT']]

    mask = prev_app.NAME_CONTRACT_STATUS == 0
    prev_app_start = prev_app.loc[mask, ['SK_ID_CURR', 'DAYS_DECISION']]

    client_employ_bureau_loan_dates = data.loc[:, ['SK_ID_CURR', 'DAYS_EMPLOYED']]\
                                           .merge(bureau_credit_start, how='left')
    client_employ_bureau_loan_dates['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    client_employ_hc_loan_dates = data.loc[:, ['SK_ID_CURR', 'DAYS_EMPLOYED']]\
                                            .merge(prev_app_start, how='left')

    client_employ_bureau_loan_dates.loc[:, 'diff_loan_before_employment'] = (client_employ_bureau_loan_dates.DAYS_CREDIT - client_employ_bureau_loan_dates.DAYS_EMPLOYED)
    client_employ_hc_loan_dates.loc[:, 'diff_loan_before_employment'] = (client_employ_hc_loan_dates.DAYS_DECISION - client_employ_hc_loan_dates.DAYS_EMPLOYED)

    btmp = client_employ_bureau_loan_dates.groupby('SK_ID_CURR')['diff_loan_before_employment'].sum()
    htmp = client_employ_hc_loan_dates.groupby('SK_ID_CURR')['diff_loan_before_employment'].sum()
    tmp  = btmp + htmp

    data.loc[:, 'sum_bureau_prev_app_credit_start_employ_start'] = data.SK_ID_CURR.map(tmp)

    del bureau_credit_start, prev_app_start, client_employ_bureau_loan_dates
    del client_employ_hc_loan_dates, btmp, htmp, tmp
    gc.collect()

    # total debt(bureau + prev_app) and total debt by income
    mask = (bureau.CREDIT_ACTIVE == 0)
    total_bureau_debt = bureau.loc[mask, :].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum()

    mask = (prev_app.NAME_CONTRACT_STATUS == 0) & (prev_app.DAYS_TERMINATION > 0) & (prev_app.CNT_PAYMENT > 0) &\
           (-prev_app.DAYS_DECISION / 30 < prev_app.CNT_PAYMENT)

    remaining_months = prev_app.loc[mask].CNT_PAYMENT - (-prev_app.loc[mask].DAYS_DECISION / 30)
    prev_app_debt    = prev_app.loc[mask].AMT_ANNUITY * remaining_months

    total_debt = total_bureau_debt.add(prev_app_debt, fill_value=0)
    data.loc[:, 'total_bureau_prev_app_live_debt'] = (data.SK_ID_CURR.map(total_debt))
    data.loc[:, 'total_bureau_prev_app_live_debt_to_income'] = data.total_bureau_prev_app_live_debt / data.AMT_INCOME_TOTAL

    # total credit
    mask = (bureau.CREDIT_ACTIVE == 0)
    total_bureau_credit = bureau.loc[mask, :].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum()

    mask = (prev_app.NAME_CONTRACT_STATUS == 0) & (prev_app.DAYS_TERMINATION > 0) & (prev_app.CNT_PAYMENT > 0) &\
        (-prev_app.DAYS_DECISION / 30 < prev_app.CNT_PAYMENT)

    remaining_months = prev_app.loc[mask].CNT_PAYMENT - (-prev_app.loc[mask].DAYS_DECISION / 30)
    prev_app_credit = prev_app.loc[mask].AMT_ANNUITY * remaining_months

    total_credit = total_bureau_credit.add(prev_app_credit, fill_value=0)
    total_credit = (data.SK_ID_CURR.map(total_credit))

    data.loc[:, 'total_live_debt_credit'] = data.total_bureau_prev_app_live_debt.div(total_credit, fill_value=np.nan).replace([-np.inf, np.inf], np.nan)
    
    del total_bureau_debt, remaining_months, prev_app_debt
    del total_debt, total_bureau_credit, prev_app_credit
    del total_credit
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

    # remaining amount left to pay on credit of previous active applications
    res = credit_bal.loc[credit_bal.NAME_CONTRACT_STATUS == 0, :]\
                    .groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['AMT_PAYMENT_TOTAL_CURRENT'].sum()

    ss  = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0) &\
                       ((prev_app.DAYS_TERMINATION > 0) | (prev_app.DAYS_TERMINATION.isnull()))
                       , ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT']]

    zz  = ss.merge(res, how='left')
    zz.loc[:, 'remaining_amount'] = zz.AMT_CREDIT - zz.AMT_PAYMENT_TOTAL_CURRENT
    pp = zz.groupby('SK_ID_CURR')['remaining_amount'].sum()
    data.loc[:, 'remaining_amount'] = data.SK_ID_CURR.map(pp)
    data.loc[:, 'diff_rem_amount_income'] = data.remaining_amount - data.AMT_INCOME_TOTAL

    del res, ss, zz, pp
    gc.collect()

    # difference in transaction activity

    # current year transaction activity
    res = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0), ['SK_ID_CURR', 'SK_ID_PREV']]
    res = res.merge(credit_bal.loc[credit_bal.MONTHS_BALANCE > -12, :], on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')

    res = res.loc[:, ['SK_ID_CURR', 
                    'CNT_DRAWINGS_ATM_CURRENT',
                    'CNT_DRAWINGS_CURRENT',
                    'CNT_DRAWINGS_OTHER_CURRENT',
                    'CNT_DRAWINGS_POS_CURRENT'
            ]]
    total = res.CNT_DRAWINGS_ATM_CURRENT.fillna(0) + res.CNT_DRAWINGS_CURRENT.fillna(0) +\
            res.CNT_DRAWINGS_OTHER_CURRENT.fillna(0) + res.CNT_DRAWINGS_POS_CURRENT.fillna(0)

    total = total.groupby(res.SK_ID_CURR).sum()
    t1    = data.SK_ID_CURR.map(total)

    # previous year transaction activity
    res = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0), ['SK_ID_CURR', 'SK_ID_PREV']]
    res = res.merge(credit_bal.loc[credit_bal.MONTHS_BALANCE <= -12, :], on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')

    res = res.loc[:, ['SK_ID_CURR', 
                    'CNT_DRAWINGS_ATM_CURRENT',
                    'CNT_DRAWINGS_CURRENT',
                    'CNT_DRAWINGS_OTHER_CURRENT',
                    'CNT_DRAWINGS_POS_CURRENT'
            ]]
    total = res.CNT_DRAWINGS_ATM_CURRENT.fillna(0) + res.CNT_DRAWINGS_CURRENT.fillna(0) +\
            res.CNT_DRAWINGS_OTHER_CURRENT.fillna(0) + res.CNT_DRAWINGS_POS_CURRENT.fillna(0)

    total = total.groupby(res.SK_ID_CURR).sum()
    t2    = data.SK_ID_CURR.map(total)

    data.loc[:, 'diff_in_transaction_activity'] = t1.subtract(t2, fill_value=0)
    
    del t1, t2, total, res
    gc.collect()

    # change in credit limit over time
    res = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0), ['SK_ID_CURR', 'SK_ID_PREV']]
    res = res.merge(credit_bal, on=['SK_ID_CURR', 'SK_ID_PREV'], how='left')

    min_ = res.groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['MONTHS_BALANCE'].min()\
          .rename(columns={'MONTHS_BALANCE': 'MIN_MONTH'})
    
    max_ = res.groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['MONTHS_BALANCE'].max()\
            .rename(columns={'MONTHS_BALANCE': 'MAX_MONTH'})

    old = res.merge(min_, left_on=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'],
          right_on=['SK_ID_CURR', 'SK_ID_PREV', 'MIN_MONTH'],
          how='inner'
         )

    new = res.merge(max_, left_on=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'],
            right_on=['SK_ID_CURR', 'SK_ID_PREV', 'MAX_MONTH'],
            how='inner'
            )

    old = old.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']].set_index(['SK_ID_CURR', 'SK_ID_PREV'])
    new = new.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']].set_index(['SK_ID_CURR', 'SK_ID_PREV'])
    
    tmp = old.subtract(new, fill_value=0).reset_index().groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].mean()
    data.loc[:, 'change_in_credit_limit_ot'] = data.SK_ID_CURR.map(tmp)

    del res, min_, max_, old, new, tmp
    gc.collect()

    # difference between cnt payment and actual duration
    tmp = installments.groupby(['SK_ID_CURR', 'SK_ID_PREV']).reset_index().rename(columns={0: 'actual_duration'})
    tmp = prev_app.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'CNT_PAYMENT']]\
              .merge(tmp, how='inner')

    tmp.loc[:, 'diff_cnt_payment'] = tmp.CNT_PAYMENT - tmp.actual_duration
    res = tmp.groupby('SK_ID_CURR')['diff_cnt_payment'].max()
    data.loc[:, 'max_diff_actual_planned_duration_prev_credit'] = data.SK_ID_CURR.map(res)
    
    del tmp, res
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

    t = installments.groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False).agg({
    'NUM_INSTALMENT_NUMBER': np.max,
    'AMT_PAYMENT': np.sum
    })

    t = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT', 'CNT_PAYMENT']]\
            .merge(t, how='left')

    res = (t.AMT_PAYMENT - t.AMT_CREDIT) * (t.CNT_PAYMENT - t.NUM_INSTALMENT_NUMBER)
    res = res.groupby(t.SK_ID_CURR).mean()

    data.loc[:, 'interaction_diff_credit_paid_term'] = data.SK_ID_CURR.map(res)
    
    res = t.AMT_CREDIT - t.AMT_PAYMENT
    res = res.groupby(t.SK_ID_CURR).min()

    data.loc[:, 'interaction_diff_credit_paid_term_min'] = data.SK_ID_CURR.map(res)

    del t, res
    gc.collect()

    # difference between current payment duration versus actual installment count
    mask = prev_app.NAME_CONTRACT_STATUS == 0

    x      = prev_app.loc[mask].groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['CNT_PAYMENT'].sum()
    inst_x = installments.groupby(['SK_ID_CURR', 'SK_ID_PREV']).size().reset_index().rename(columns={0: 'inst_size'})

    tmp = x.merge(inst_x)
    tmp.loc[:, 'diff'] = tmp.CNT_PAYMENT - tmp.inst_size
    tmp = tmp.groupby('SK_ID_CURR')['diff'].sum()
    data.loc[:, 'diff_actual_duration_installment_count'] = data.SK_ID_CURR.map(tmp)

    del tmp, x, inst_x
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

    # difference between latest bureau and previous application loan
    t1 = prev_app[prev_app.NAME_CONTRACT_STATUS == 0].groupby('SK_ID_CURR')['DAYS_DECISION'].min()
    t2 = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].min()

    t1 = data.SK_ID_CURR.map(t1)
    t2 = data.SK_ID_CURR.map(t2)

    data.loc[:, 'min_diff_bureau_prev_app_credit_date'] = (t1 - t2)

    del t1, t2
    gc.collect()

    t1 = prev_app[prev_app.NAME_CONTRACT_STATUS == 0].groupby('SK_ID_CURR')['DAYS_DECISION'].max()
    t2 = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].max()

    t1 = data.SK_ID_CURR.map(t1)
    t2 = data.SK_ID_CURR.map(t2)

    data.loc[:, 'max_diff_bureau_prev_app_credit_date'] = (t1 - t2)


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
    tmp = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0) &\
                       ((prev_app.DAYS_TERMINATION.isnull()) | (prev_app.DAYS_TERMINATION > 0))
                    , ['SK_ID_CURR', 'SK_ID_PREV']]
    res = pos_cash.groupby(['SK_ID_CURR', 'SK_ID_PREV'], as_index=False)['MONTHS_BALANCE'].max()
    res = res.merge(pos_cash, how='left')
    res = res.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].sum()

    data.loc[:, 'total_remaining_cash_credit_term'] = data.SK_ID_CURR.map(res)

    del res, tmp
    gc.collect()

    res = prev_app.loc[prev_app.NAME_CONTRACT_STATUS == 0, ['SK_ID_CURR',
                                                           'SK_ID_PREV',
                                                           'AMT_CREDIT',
                                                           'AMT_ANNUITY',
                                                           'RATE_DOWN_PAYMENT'
                                                          ]].merge(pos_cash.loc[:, ['SK_ID_CURR',
                                                                                    'SK_ID_PREV',
                                                                                    'CNT_INSTALMENT',
                                                                                    'CNT_INSTALMENT_FUTURE'
                                                                                   ]],
                                                                   how='left'
                                                                  )
    
    res.loc[:, 'credit_share_left']  = (res.AMT_ANNUITY * res.CNT_INSTALMENT_FUTURE * res.RATE_DOWN_PAYMENT) / res.AMT_CREDIT
    tmp                              = res.groupby('SK_ID_CURR')['credit_share_left'].mean()
    data.loc[:, 'credit_share_left'] = data.SK_ID_CURR.map(tmp)

    # Months Balance info from pos_cash table.
    m = prev_app.loc[(prev_app.NAME_CONTRACT_STATUS == 0) &\
             (prev_app.NAME_CONTRACT_TYPE != 2)
             , ['SK_ID_CURR', 'SK_ID_PREV']]\
        .merge(pos_cash.loc[:, ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE']])

    m = m.sort_values(by=['SK_ID_CURR', 'MONTHS_BALANCE'], ascending=[True, False])
    f = m.groupby('SK_ID_CURR', as_index=False)['MONTHS_BALANCE'].first()
    l = m.groupby('SK_ID_CURR', as_index=False)['MONTHS_BALANCE'].last()

    tmp = 0 - l['MONTHS_BALANCE']
    f.loc[:, 'diff'] = tmp

    tmp = data.loc[:, ['SK_ID_CURR', 'TARGET']].merge(f.loc[:, ['SK_ID_CURR', 'diff']], on='SK_ID_CURR', how='left')
    data.loc[:, 'pos_current_last_months_balance'] = tmp['diff']

    del tmp, m, l, f
    gc.collect()


    return data, list(set(data.columns) - set(COLS))

def prev_app_pos_credit(prev_app, pos_cash, credit_bal, data):
    COLS = data.columns.tolist()

    # total days past due
    mask = (prev_app.NAME_CONTRACT_STATUS == 0) &\
           (prev_app.DAYS_DECISION >= -365)

    p  = prev_app.loc[mask , ['SK_ID_PREV', 'SK_ID_CURR']]
    c  = p.merge(pos_cash, how='inner')
    cc = p.merge(credit_bal, how='inner')

    # SUM
    c_dpd  = c.groupby(['SK_ID_CURR'])['SK_DPD'].sum()
    cc_dpd = cc.groupby(['SK_ID_CURR'])['SK_DPD'].sum() 

    data.loc[:, 'cash_dpd_sum']   = data.SK_ID_CURR.map(c_dpd)
    data.loc[:, 'credit_dpd_sum'] = data.SK_ID_CURR.map(cc_dpd) 

    res = c_dpd.add(cc_dpd, fill_value=np.nan)
    res = data.SK_ID_CURR.map(res)

    data.loc[:, 'total_cash_credit_dpd'] = res.fillna(-1).astype(np.int8)
    
    # Std
    c_dpd  = c.groupby(['SK_ID_CURR'])['SK_DPD'].std()
    cc_dpd = cc.groupby(['SK_ID_CURR'])['SK_DPD'].std() 

    data.loc[:, 'cash_dpd_std']   = data.SK_ID_CURR.map(c_dpd)
    data.loc[:, 'credit_dpd_std'] = data.SK_ID_CURR.map(cc_dpd) 

    # Max
    c_dpd  = c.groupby(['SK_ID_CURR'])['SK_DPD'].max()
    cc_dpd = cc.groupby(['SK_ID_CURR'])['SK_DPD'].max() 

    data.loc[:, 'cash_dpd_max']   = data.SK_ID_CURR.map(c_dpd)
    data.loc[:, 'credit_dpd_max'] = data.SK_ID_CURR.map(cc_dpd) 

    del p, c, cc, c_dpd, cc_dpd, res
    gc.collect()

    return data, list(set(data.columns) - set(COLS))

def prev_app_ohe(prev_app, data):
    COLS = data.columns.tolist()

    # NAME_CONTRACT_TYPE
    tmp         = prev_app.groupby(['SK_ID_CURR', 'NAME_CONTRACT_TYPE']).size().unstack().fillna(0).reset_index()
    tmp.columns = [f'NAME_CONTRACT_TYPE{col}' if col != 'SK_ID_CURR' else col for col in tmp.columns]
    
    data        = merge(data, tmp)

    del tmp
    gc.collect()

    # NAME_CONTRACT_STATUS
    tmp         = prev_app.groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).size().unstack().fillna(0).reset_index()
    tmp.columns = [f'NAME_CONTRACT_STATUS_{col}' if col != 'SK_ID_CURR' else col for col in tmp.columns]
    
    data        = merge(data, tmp)

    del tmp
    gc.collect()

    # CODE REJECT REASON
    tmp         = prev_app.groupby(['SK_ID_CURR', 'CODE_REJECT_REASON']).size().unstack().fillna(0).reset_index()
    tmp.columns = [f'CODE_REJECT_REASON_{col}' if col != 'SK_ID_CURR' else col for col in tmp.columns]
    
    data        = merge(data, tmp)

    del tmp
    gc.collect()

    # PRODUCT COMBINATION
    tmp         = prev_app.groupby(['SK_ID_CURR', 'PRODUCT_COMBINATION']).size().unstack().fillna(0).reset_index()
    tmp.columns = [f'PRODUCT_COMBINATION_{col}' if col != 'SK_ID_CURR' else col for col in tmp.columns]
    
    data        = merge(data, tmp)

    del tmp
    gc.collect()

    # NAME_YIELD_GROUP
    tmp         = prev_app.groupby(['SK_ID_CURR', 'NAME_YIELD_GROUP']).size().unstack().fillna(0).reset_index()
    tmp.columns = [f'NAME_YIELD_GROUP_{col}' if col != 'SK_ID_CURR' else col for col in tmp.columns]
    
    data        = merge(data, tmp)

    del tmp
    gc.collect()

    # NAME_GOODS_CATEGORY
    tmp         = prev_app.groupby(['SK_ID_CURR', 'NAME_GOODS_CATEGORY']).size().unstack().fillna(0).reset_index()
    tmp.columns = [f'NAME_GOODS_CATEGORY_{col}' if col != 'SK_ID_CURR' else col for col in tmp.columns]
    
    data        = merge(data, tmp)

    del tmp
    gc.collect()

    # WEEKDAY_APPR_PROCESS_START
    tmp         = prev_app.groupby(['SK_ID_CURR', 'WEEKDAY_APPR_PROCESS_START']).size().unstack().fillna(0).reset_index()
    tmp.columns = [f'WEEKDAY_APPR_PROCESS_START_{col}' if col != 'SK_ID_CURR' else col for col in tmp.columns]
    
    data        = merge(data, tmp)

    del tmp
    gc.collect()

    # HOUR_APPR_PROCESS_START
    tmp         = prev_app.groupby(['SK_ID_CURR', 'HOUR_APPR_PROCESS_START']).size().unstack().fillna(0).reset_index()
    tmp.columns = [f'HOUR_APPR_PROCESS_START_{col}' if col != 'SK_ID_CURR' else col for col in tmp.columns]
    
    data        = merge(data, tmp)

    del tmp
    gc.collect()

    # CHANNEL_TYPE
    tmp         = prev_app.groupby(['SK_ID_CURR', 'CHANNEL_TYPE']).size().unstack().fillna(0).reset_index()
    tmp.columns = [f'CHANNEL_TYPE_{col}' if col != 'SK_ID_CURR' else col for col in tmp.columns]
    
    data        = merge(data, tmp)

    del tmp
    gc.collect()

    return data, list(set(data.columns) - set(COLS))