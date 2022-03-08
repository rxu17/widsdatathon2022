# -*- coding: utf-8 -*-
'''
Description: Runs data models

Arguments: 
    

How To Use: python 
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

import model_selection as model_select
from utils.tests import *
from b_data_prep import run_data_prep


def run_models():
    '''
    '''

def main(args):
    ''' Go in order
    '''
    data_prep_params = {'selected_vars':['avg_temp', 'non_tmp'],
                        'transform_vars': ['building_class', 'facility_type', 'State_Factor'], 
                        'transform_methods':['one_hot_encode', 'one_hot_encode', 'one_hot_encode'],
                        'outlier_vars':['site_eui'], 'outlier_methods':['capping'],
                        'impute_vars':['energy_star_rating'], 'imputer_methods':['knn']}
    
    prepped_train, prepped_valid, prepped_test = run_data_prep.main(data_prep_params)
    import pdb; pdb.set_trace()
    for model in ["decision", "randomforest", "linear", "ada"]:
         model_select.run(train = prepped_train, valid = prepped_valid, model = model,
                    best_features = ['snowfall_inches', 'direction_max_wind_speed',
                                    'precipitation_inches', 'days_above_110F', 'snowdepth_inches',
                                    'days_above_80F', 'days_above_90F', 'days_above_100F',
                                    'id', 'avg_temp', 'Year_Factor', 'floor_area', 'year_built', 'ELEVATION',
                                    'january_avg_temp', 'february_avg_temp', 'march_avg_temp',
                                    'april_avg_temp', 'may_avg_temp', 'june_avg_temp', 'july_avg_temp',
                                    'august_avg_temp', 'september_avg_temp', 'october_avg_temp',
                                    'november_avg_temp', 'december_avg_temp', 'energy_star_rating'])


if __name__ == "__main__":
    main(sys.argv)