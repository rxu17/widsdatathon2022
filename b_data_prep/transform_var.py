# -*- coding: utf-8 -*-
'''
Description: Used to transform icu health data, such as converting categorical 
data into numeric, or creating new indicators

# convert days_below_30F to days_below_30F and days_above_20F, by method of subtracting days_above_20F - days_above_30F
# could change days to a percentage so there is a "bound" on the number (scaling it almost out of 100)

# factorize building_class, and facility_type (try two methods for these), 
# make them into factors or make them into dummy variables and see if correlation improves

Contents:
How To Use:
Contributors: rxu17
'''

import re
import time
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn import preprocessing


PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from utils.tests import *


def one_hot_encoding(input_df, trans_var):
    ''' This function converts each category value into a new column and 
        assigns a 1 or 0 (True/False) value to the column.

        Parameters:
            input_df: dataframe with object columns

        Returns: df with all possible val in each object column 
        encoded as a true/false columns
    '''
    encoded_df = pd.get_dummies(input_df, columns=[trans_var], 
                                prefix = "dum_{}".format(trans_var))
    return(encoded_df)


def label_encoding(input_df, trans_var):
    ''' Encodes the column into numbers (useful for ordinal columns)

        Parameters:
            trans_var: list of cols in df to encode
            input_df: dataframe of our input_df
        
        Returns: df with selected variables encoded
    '''
    input_df[trans_var] = input_df[trans_var].astype('category')
    input_df[trans_var] = input_df[trans_var].cat.codes
    return(input_df)


def variable_binning(input_df, bin_info):
    ''' For when we want to bin variables into different groups whether for 
        better estimates when we have too many categories or cap creation

        Paramaters:
            input_df: our input df for variable binning
            bin_info: dict of keys being bin cols and values being bin widths

        Returns: df with selected variables binned
    '''
    assert(isinstance(bin_info, 'dict') & set(bin_info.keys()) <= set(input_df.columns)), \
        "Error: bin_info is not a dictionary object and/or columns to bin doesn't exist in input_df"
    for col in bin_info.keys(): # loop through all variables to bin
        assert(bin_info[col] < max(input_df[col]) - min(input_df[col])), \
            "bin width of {wid} for column {col} is too big".format(wid = bin_info[col], col = col)
        bins = np.arange(math.floor(min(input_df[col])), math.floor(max(input_df[col])), 
                                                                            bin_info[col]) 
        labels = list(range(1, len(bins)))
        input_df['{}_bin'.format(col)] = pd.cut(input_df[col], bins=bins, labels=labels)
    return(input_df)


def variable_scaling(input_df, scale_var, use_outlier_scaling):
    ''' Scaling variables (mean of 0, sd of 1) for better handling by models

        Paramaters:
            input_df: our input df for variable scaling
            scale_vars: dict of keys being bin cols and values being bin widths
            use_outlier_scaling: [T,F] whether dataset contains outliers

        Returns: df with selected variables scaled
    '''
    if use_outlier_scaling:
        scaled_df = preprocessing.robust_scale(input_df[scale_var])
    else:
        scaled_df = preprocessing.scale(input_df[scale_var])
    return(scaled_df)


def run(input, trans_vars, transformer, is_test : bool = False):
    '''
    '''
    assert set(transformer) <= set(get_model_param("transformers"))
    if trans_vars is None and transformer is None:
        return(input)
    
    # call transformer
    trans_dict = {trans_vars[i]:transformer[i] for i in range(len(trans_vars))}
    updated_df = input.copy()
    print("Running transformer ...")
    for trans_var, transformer in trans_dict.items():
        if is_test and set([trans_var]) == set(get_model_param("target_var")):
            trans_vars = list(set(trans_vars) - set([trans_var]))
            continue
        if transformer == "one_hot_encode":
            updated_df = one_hot_encoding(updated_df, trans_var)
        elif transformer == "label_encode":
            updated_df = label_encoding(updated_df, trans_var)
        elif transformer == "var_binning":
            updated_df = variable_binning(updated_df, 
                                          bin_info = get_model_param("var_bin_info"))
        elif transformer == "var_scaling":
            updated_df = variable_scaling(updated_df, trans_var, 
                                          use_outlier_scaling=get_model_param("use_outlier_scaling"))

    print("Values are transformed...")
    new_trans_vars = [
        col for col in updated_df 
        if col in trans_vars or col.startswith("dum_")
    ]
    test_zero_inf_null_vals(table = updated_df, vars = new_trans_vars)
    print("Passed zero, infinity, null values check!")
    return(updated_df)
