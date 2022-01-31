'''
Description: This script searches for missingness in a dataset and based on
               user selection, takes an imputation method and fills in the missingness

Arguments: None
How To Use: import 
Contributors: rxu17
'''
import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
#from sklearn.impute import IterativeImputer

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from utils.tests import *


def run(input : pd.DataFrame, impute_vars : list = None, 
        imputer : list = None) -> pd.DataFrame:
    ''' Select imputation method:
            knn - k nearest neighbors
            mode - fill with frequent value
            hot deck -
            linear reg - 
    '''
    assert set(imputer) <= set(get_model_param("imputers"))
    if impute_vars is None and imputer is None:
        return(input)
    
    # call imputer
    impute_dict = {impute_vars[i]:imputer[i] for i in range(len(impute_vars))}
    updated = input.copy()
    print("Running imputer ...")
    for impute_var, imputer in impute_dict.items():
        if imputer == "knn":
            helper_vars = get_model_param("knn_vars")
            knn = KNNImputer(n_neighbors=5, weights="uniform")
            knn_vals = knn.fit_transform(updated[helper_vars + [impute_var]].values)
            updated_new = pd.DataFrame(knn_vals, index=updated.index, columns=helper_vars + [impute_var])
            updated = updated.drop(helper_vars + [impute_var], axis=1)
            updated = pd.concat([updated, updated_new], axis=1)
        elif imputer == "mode":
            mode = input[impute_var].mode()
            updated[impute_var] = updated[impute_var].fillna(mode)
        elif imputer == "hot_deck":
            updated[impute_var] = updated[impute_var].fillna(method='ffill')
        elif imputer == "linear_reg":
            updated[impute_var]= updated[impute_var].interpolate(
                                            method='linear', 
                                            limit_direction="both")
    print("Values are imputed...")
    test_zero_inf_null_vals(table = updated, vars = impute_vars)
    print("Passed zero, infinity, null values check!")
    return(updated)
