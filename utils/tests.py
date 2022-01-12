
'''
Description: General tests to use around pipeline
How To Use: 
Contributors: rxu17
'''

import os
import sys
import yaml
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


def test_var_in_table(table : pd.DataFrame, vars : list) -> None:
    ''' Check if all variables are present in table
    '''
    vars_in_tab = list(table.columns)
    assert set(vars) <= set(vars_in_tab),\
        "Missing variables in table:\n {}".format(
            list(set(vars) - set(vars_in_tab)))


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