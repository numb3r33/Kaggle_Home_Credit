import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def fill_missing_values(data):
    for col in data.columns:
        # replace inf with np.nan
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)

        # fill missing values with median
        if data[col].isnull().sum():
            if pd.isnull(data[col].median()):
                data[col] = data[col].fillna(-1)
            else:
                data[col] = data[col].fillna(data[col].median())

    return data