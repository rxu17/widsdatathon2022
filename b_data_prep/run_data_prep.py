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

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

# import data prep steps
import b_data_prep.select_var as select
import b_data_prep.impute_missing as impute
import b_data_prep.remove_outliers as remove_out
import b_data_prep.transform_var as trans_var
# for splitting data
from sklearn.model_selection import train_test_split
from utils.tests import *


def read_and_format_data():
    ''' Reads and splits training data
    '''
    def _validate_path(data_type):
        ''' check data path
        '''
        input_path = get_path(key = "{}_data".format(data_type), 
                              pipeline_step = "data_intake")
        test_filepath(input_path)
        test_file(input_path)
        return(input_path)

    # split data
    input_df = pd.read_csv(_validate_path(data_type="train"))
    validate_size = get_model_param("split_size")
    train_data, validation_data = \
        train_test_split(input_df, test_size=validate_size, random_state=6)

    test_data = pd.read_csv(_validate_path(data_type="test"))
    return(train_data, validation_data, test_data)


def validate_args():
    '''
    '''

def main_test(args):
    ''' Go in order
    '''
    input_path = get_path(key = "{}_data".format(args[0]), pipeline_step = "data_intake")
    if args[0] == "train":
        input_df, valid_df = read_and_format_data(input_path, data_type = args[0])
    elif args[0] == "test":
        input_df = read_and_format_data(input_path, data_type = args[0])
    args_test = ['avg_temp', 'non_tmp']
    selected = select.run(input = input_df, selected_vars = args_test)#args[0])

    args_test2 = ['energy_star_rating']
    args_test3 = ['knn']

    args_test4 = ['site_eui']
    args_test5 = ['capping']

    args_test6 = ['building_class', 'facility_type', 'State_Factor']
    args_test7 = ['one_hot_encode', 'one_hot_encode', 'one_hot_encode']
    removed = remove_out.run(input = selected, out_vars = args_test4, out_remover = args_test5)
    imputed = impute.run(input = removed, impute_vars = args_test2, imputer = args_test3)
    transformed = trans_var.run(input = imputed, trans_vars= args_test6, transformer= args_test7)
    return(transformed)


def run_data_prep(input_df, args, is_test):
    ''' Run the steps
    '''
    selected = select.run(input = input_df, 
                          selected_vars = args['selected_vars'], is_test = is_test)
    removed = remove_out.run(input = selected, out_vars = args['outlier_vars'], 
                             out_remover = args['outlier_methods'], is_test = is_test)
    imputed = impute.run(input = removed, impute_vars = args['impute_vars'], 
                         imputer = args['imputer_methods'], is_test = is_test)
    transformed = trans_var.run(input = imputed, trans_vars= args['transform_vars'], 
                                transformer= args['transform_methods'], is_test = is_test)
    return(transformed)


def main(args):
    ''' Run for train, validation and test datasets
    '''
    train_df, valid_df, test_df = read_and_format_data()
    prepped_train = run_data_prep(train_df, args, is_test = False)
    prepped_valid = run_data_prep(valid_df, args, is_test = False)
    prepped_test = run_data_prep(test_df, args, is_test = True)
    return(prepped_train, prepped_valid, prepped_test)
