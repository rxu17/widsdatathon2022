'''
Description: This script models

Can try ensemble modeling where 100 models are generated and averaged

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

def detect_outliers(input : pd.DataFrame, out_var : list, 
                    detect_method : str) -> pd.DataFrame:
    ''' iqr method (Q1 - 1.5* IQR) (Q3 + 1.5* IQR)
    '''
    print("Preparing for outlier detecting ...")
    if detect_method == "iqr":
        # IQR
        Q1 = np.percentile(input[out_var], 25,
                        interpolation = 'midpoint')
        Q3 = np.percentile(input[out_var], 75,
                        interpolation = 'midpoint')
        IQR = Q3 - Q1
        input['is_upper_out'] = np.where(input[out_var] >= (Q3+1.5*IQR), 1, 0)
        input['is_lower_out'] = np.where(input[out_var] <= (Q1-1.5*IQR), 1, 0)
        input['is_out'] = np.where((input['is_upper_out']==1)|(input['is_lower_out']==1), 1, 0)
    return(input)
        


def create_and_apply_caps(input_df, cap_type, out_var):
    '''
    '''
    print("Creating caps ...")
    # define cap quantiles
    cap_type_val = {"upper_cap":0.975, "lower_cap":0.025}
    cap_group = get_model_param("cap_group")
    caps = input_df.groupby(cap_group).quantile(cap_type_val[cap_type]).reset_index()
    # merge in caps
    caps.rename({out_var:"{}_{}".format(out_var, cap_type)}, axis = 1, inplace = True)
    input_w_cap = input_df.merge(caps[["{}_{}".format(out_var, cap_type)] + cap_group], 
                                  on = cap_group, how = "left")
    # apply caps
    print("Applying caps ...")
    out_type = cap_type.split("_")[0]
    input_w_cap[out_var] = np.where(input_w_cap['is_{}_out'.format(out_type)]==1, 
                            input_w_cap["{}_{}".format(out_var, cap_type)], input_w_cap[out_var])
    input_w_cap = input_w_cap.drop("{}_{}".format(out_var, cap_type), axis = 1)
    return(input_w_cap)


def run(input : pd.DataFrame, out_vars : list = None, 
        out_remover : list = None) -> pd.DataFrame:
    ''' Select imputation method:
            capping - takes 95% and 5% quntiles of our data by group of vars, and 
                          caps our data at those values
            removal - removes those rows completely
            NA - converts all rows to NA
    '''
    assert set(out_remover) <= set(get_model_param("outlier_removers"))
    if out_vars is None and out_remover is None:
        return(input)
    # call outlier remover
    out_dict = {out_vars[i]:out_remover[i] for i in range(len(out_vars))}
    updated = input.copy()
    for out_var, out_remover in out_dict.items():
        updated = detect_outliers(updated, out_var,detect_method="iqr")
        if out_remover == "capping":
            updated = create_and_apply_caps(updated, "upper_cap", out_var)
            updated = create_and_apply_caps(updated, "lower_cap", out_var)
            updated = updated.drop(['is_out', 'is_lower_out', 'is_upper_out'], axis = 1)
        elif out_remover == "NA":
            updated.loc[updated['is_out']==1, out_var] = np.NaN
        elif out_remover == "removal":
            updated = updated.loc[updated['is_out']==0]
    
    print("Outlier removal completed!")
    test_zero_inf_null_vals(table = updated, vars = out_vars)
    return(updated)
