# -*- coding: utf-8 -*-
'''
Description: Runs data preparing on the dataset
    - removes redundant variables given by selection
        # new variable of avg_temp, as keeping min_temp, avg_temp and max_temp is too redundant

        # avg_temp vars and the _degree_days and below_<some temp in F> vars are all redundant, 
        # so we have to keep one set or the other

Arguments: None
How To Use: import select_var
Contributors: rxu17
'''
import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from utils.tests import *


def run(input : pd.DataFrame, selected_vars : list, is_test : bool = False) -> pd.DataFrame:
    ''' Special key words for selected_vars
        selected_vars:
            Allowed values: variables in input, 
            ['avg_temp','min_temp', 'max_temp', '_degree', 'below_', 'all', 'non_tmp']
            avg_temp, min_temp, max_temp, _degree, below_ - only select variables with this substring
            all - selected all available variables
            non_tmp - selects all non temperature related variables
        is_test: whether this is test data or not
    '''
    spe_vars =  ['avg_temp','min_temp', 'max_temp', '_degree', 'below_', 'all', 'non_tmp']
    tmp_vars = ['avg_temp','min_temp', 'max_temp', '_degree', 'below_']
    for spe_var in spe_vars:
        if spe_var in selected_vars:
            selected_vars.remove(spe_var)
            if spe_var == "all": # include all variables
                selected_vars = list(input.columns)
            elif spe_var == "non_tmp":
                selected_vars += [var for var in input.columns 
                                    if '_temp' not in var 
                                    and '_degree' not in var 
                                    and 'below_' not in var]
            else: # include only subset based on specified prefix/suffix
                selected_vars += [var for var in input.columns if spe_var in var]
    
    # reattach the target var variable and id vars
    target_var = get_model_param("target_var") if not is_test else []
    id_vars = get_model_param("id_vars")
    selected_vars = list(set(selected_vars) - set(id_vars + get_model_param("target_var")))
    test_var_in_table(table = input, vars = list(set(selected_vars)))

    input = input[list(set(selected_vars + id_vars + target_var))]
    return(input)
