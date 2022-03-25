
'''
Description: General tests to use around pipeline
How To Use: 
Contributors: rxu17
'''

import os
import sys
import yaml
import numpy as np
import pandas as pd

def get_path(key : str, pipeline_step : str) -> str:
    '''
    '''
    # Load the YAML file
    with open('{}/data_paths.yaml'.format(os.path.dirname(__file__))) as fh:
        # Convert the YAML data into a dictionary
        dictionary_data = yaml.safe_load(fh)
    return("{}/{}".format(os.path.abspath(os.curdir), 
                          dictionary_data[pipeline_step][key]))


def get_model_param(key : str) -> str:
    '''
    '''
    # Load the YAML file
    with open('{}/model_parameters.yaml'.format(os.path.dirname(__file__))) as mp:
        # Convert the YAML data into a dictionary
        dictionary_data = yaml.safe_load(mp)
    return(dictionary_data[key])


def test_var_in_table(table : pd.DataFrame, vars : list) -> None:
    ''' Check if all variables are present in table
    '''
    vars_in_tab = list(table.columns)
    assert set(vars) <= set(vars_in_tab),\
        "Missing variables in table:\n {}".format(
            list(set(vars) - set(vars_in_tab)))


def test_zero_inf_null_vals(table : pd.DataFrame, vars : list) -> None:
    ''' Check if any variables have any nulls, inf values or all 0s
    '''
    vars_in_tab = list(table.columns)
    test_dict = {}
    for var in vars:
        test_dict[var] = {}
        # check for all 0s
        test_dict[var]['has_all_zeroes'] = (table[var] == 0).all()
        test_dict[var]['has_nas'] = table[var].isnull().values.any()
        test_dict[var]['has_inf'] = (abs(table[var] == np.inf)).values.any()
        if not True in test_dict[var].values():
            del test_dict[var]

    assert len(list(test_dict.keys())) == 0,\
        "Dataframe has one or more of the following:\n{}".format(
            test_dict
        )


def test_filepath(filepath : str) -> None:
    ''' series of tests for filepath:
        -Check for read access
        -Check for write access
        -Check for execution access'''
    assert(os.path.isdir(filepath), "Dir doesn't exist")
    assert(os.access(filepath, os.R_OK), "Dir not readable") 
    assert(os.access(filepath, os.W_OK), "Dir not writeable")
    assert(os.access(filepath, os.X_OK), "Dir not executable")


def test_file(file):
    ''' series of tests for file, check it exists and is writable/readable'''
    assert(os.path.isfile(file), "File:{} doesn't exist".format(file))
    try:
        f = open(file)
        f.close()
    except IOError:
        print("File:{} not accessible".format(file))
    finally:
        pass