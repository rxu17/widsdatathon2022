# -*- coding: utf-8 -*-
'''
Description: Used to model energy eui
Contents: 
Arguments: main_process_ids 
           remove_outliers_ids
           impute_missing_ids
           transform_type_ids
           variable_type_ids
           model_type_ids
How To Use: python launch_survival_pipeline.py -mp <main_process_ids>
                   -process_ids <process_ids> -reout_ids <remove_outlier_ids>
                   -imp_ids <imputation_ids> -transf_ids <transformation_ids>
                   -var_ids <variable selection ids> -model_ids <model_type_ids>
Contributors: rxu17
'''

import re
import time
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import seaborn as sb
import yaml
from getpass import getuser


def get_pipeline_process() -> dict:
    '''
    '''
    steps = {1: {"data_intake" : []},
             2: {"data_prep" : ['reout_ids', 'transf_ids', 'imp_ids']},
             3: {"modeling" : ['var_ids', 'model_ids']},
             4: {"model_evaluation" :[]},
             5: {"model_deployment":{]}}}

def get_current_repo() -> str:
    ''' 
    Returns the repo location for the code that is currently running.
    '''
    try:
        this_dir = os.path.realpath(__file__).strip(".py")
    except Exception:
        this_dir = os.getcwd()
        assert '/wids_datathon_patient_icu' in this_dir, \
            "ERROR: code run interactively must be run from a repo sub-directory"
    repo_len = len("/wids_datathon_patient_icu") + 1
    end_repo_name = this_dir.find("/wids_datathon_patient_icu")+repo_len
    code_repo = this_dir[:end_repo_name].rstrip(
        '/')  # slice and remove trailing slash
    return(code_repo)


def get_path(main_process = None, step = None) -> str:
    ''' Gets the full filepath for a given process name and step
        
        Args: 
            main_process - str, see data_paths.yaml for possible 
                    values, process names
            step - str, see data_paths.yaml for possible 
                    values, step names
    '''
    path_file = get_current_repo() + "/data_paths.yaml"
    with open(path_file) as data_file:
        parameter_dict = yaml.load(data_file)
    if main_process not in list(parameter_dict.keys()):
        raise LookupError("Could not find specified main process, "
                          "{}, in list of parameters".format(main_process))
    if step is None:
        pass
    if step not in list(parameter_dict[main_process].keys()):
        raise LookupError("Could not find specified step, "
                          "{}, in list of parameters".format(step))
    return(parameter_dict[main_process][step])


def launch(process_ids : list, reout_ids : list, imp_ids : list,
           trans_ids : int, var_ids : int, model_ids : int) -> None
    ''' Launches the different scripts for your data process

        Args: process_ids - input 
              reout_ids - 
              imp_ids -
              transf_ids -
              var_ids -
              model_ids -

    '''
    pp = get_pipeline_process()
    for process_id in process_ids:
        args = []
        script = get_path(pp[process_id].keys())
        cmd = 'python {} {}'.format(script, args.split(" "))
        subprocess.call(cmd, shell = False)


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run specified processes')
    parser.add_argument('-mp', '--main_process_ids', type=int,
                        default=[1, 2, 3, 4, 5],
                        nargs='*',
                        help=('List of main processes you wish to run'))
    parser.add_argument('-reout_ids', '--remove_outliers_ids', type=int,
                        default=[1, 2, 3],
                        nargs='?',
                        help=('Type of outlier removal process'))
    parser.add_argument('-imp_ids', '--impute_missing_ids', type=int,
                        default=[1, 2, 3],
                        nargs='?',
                        help=('Type of imputation process on missing data'))
    parser.add_argument('transf_ids', '--transform_type_ids', type=int,
                    default=[1, 2, 3],
                    nargs='?',
                    help=('Type of transformation to run, otherwise runs for all'))
    parser.add_argument('-var_ids', '--variable_type_ids', type=int,
                        default=[1, 2, 3],
                        nargs='?',
                        help=('Type of variable selection method to run, otherwise runs for all'))
    parser.add_argument('-model_ids', '--model_type_ids', type=int,
                        default=[1, 2, 3],
                        nargs='?',
                        help=('Type of model to run, otherwise runs for all'))
    return(parser)

def main():
    args = get_args()
    launch(args.main_process_ids, 
           args.remove_outliers_ids, args.impute_missing_ids,
           args.transform_type_ids, args.variable_type_ids,
           args.model_type_ids)


if __name__ == "__main__":
    main()
