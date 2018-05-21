import numpy as np
import pandas as pd

def woe(df, tr, f, BINS=10):
    """
    For continuous variables only.
    """
    f_with_missing = pd.factorize(df[f])[0]
    f_binned       = pd.cut(f_with_missing, bins=BINS, labels=np.arange(BINS))

    grp       = tr.groupby([f_binned[:len(tr)], 'TARGET']).size().unstack().fillna(0)
    grp_share = grp / grp.sum()

    woe = grp_share.apply(lambda x: np.log(x[0.0] / x[1.0]), axis=1)
    iv  = grp_share.apply(lambda x: x[0] - x[1], axis=1) * woe.values

    return f_binned, woe, iv.sum()