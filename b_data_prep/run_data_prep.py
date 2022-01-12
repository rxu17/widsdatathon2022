# -*- coding: utf-8 -*-
'''
Description: Runs data preparing on the dataset prior to any modeling or feature selection
    - removes redundant variables/selects subset of variables
    - transforms data
    - remove outliers
    - imputes missingness

Arguments: 
    selected_vars - str of variables to keep
    impute_vars - str, variables to perform imputation on
    imputer - str, imputation method(s), if list, must be same length as impute_vars
    out_vars - str, variables to perform outlier removal on
    out_remover - str, outlier removal methods, if list, must be same length as out_vars
    trans_vars - str, variables to perform transformations on
    transformer - str, transformation methods, if list, must be same length as trans_vars

How To Use: python run_data_prep.py <selected_vars> <imput_vars> <imputer> <out_vars> 
                                <out_remover> <trans_vars> <transformer>
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

# import data prep steps
import select_var as select
import impute_missing as impute
#import remove_outliers as remove_out
import transform_var as trans_var

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from utils.tests import *

def read_and_format_data(input_path):
    test_filepath(input_path)
    test_file(input_path)
    input_df = pd.read_csv(input_path)
    return(input_df)

def validate_args():
    '''
    '''

def main(args):
    ''' Go in order
    '''
    input_path = get_path(key = "train_data", pipeline_step = "data_intake")
    input_df = read_and_format_data(input_path)
    args_test = ['avg_temp', 'non_tmp']
    selected = select.run(input = input_df, selected_vars = args_test)#args[0])
    imputed = impute.run(input = selected, impute_vars = args[1], imputer = args[2])
    removed = remove_out.run(input = imputed, out_vars = args[3], out_remover = args[4])
    transformed = trans_var.run(input = removed, trans_vars= args[4], transformer= args[5])

if __name__ == "__main__":
    main(sys.argv)