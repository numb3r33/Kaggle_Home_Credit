import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

def create_folds(data, seed):
    dtr, dte, _, _  = train_test_split(data, data.TARGET, stratify=data.TARGET, test_size=.3, random_state=seed)
    return dtr, dte