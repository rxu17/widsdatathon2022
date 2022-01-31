# -*- coding: utf-8 -*-
'''
Description: This script is used to test out various 
variable selection methods for the icu patient survival model
Contents:
How To Use:
Contributors: rxu17
'''

import time
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from utils.tests import *

def find_best_model_params(model : object, train_feat : pd.DataFrame,
                           train_target : pd.DataFrame, hyperparams : dict,
                           cv : int) -> dict:
    ''' Parameters:
        ================
        model : selected model
        train_feat: features for use
        train_target: our target var
        hyper_params : dictionary of the parameters with possible values by model
        cv : parameter of Grid Search
        
        Returns: model parameters
    '''
    if isinstance(train_feat, pd.DataFrame):
        train_df = train_feat.to_numpy()
    else:
        train_df = train_feat.copy()
    search = GridSearchCV(estimator = model,
                          param_grid=hyperparams,
                          return_train_score=True,
                          cv = cv).fit(X = train_df, 
                                       y = train_target.to_numpy())
    return(search.best_params_)



def train_and_evaluate_model(model : object,
                             model_type : str,
                             X_train : pd.DataFrame, y_train : pd.DataFrame, 
                             X_valid : pd.DataFrame, y_valid : pd.DataFrame,
                             model_params : dict,
                             run_test : bool = False, 
                             X_test : pd.DataFrame = None, 
                             y_test : pd.DataFrame = None,
                             return_pred : bool = False) -> pd.DataFrame:
    ''' Parameters:
        =============
        model_type - Allowed values: ['decision', "randomforest",'adaboost']
        model_params - input parameters for specific model of your choice
        X_train - training input data with selected features
        y_train - training data with target variable
        X_valid - validation input data with selected features
        y_valid - validation data with target variable
        run_test - whether to run model prediction for testing data, Allowed Values: [True, False]
        X_test - testing input data with selected features, Optional if run_test is False
        y_test - testing data with target variable, Optional if run_test is False
        
        Returns: root mean square error of the predicted target values with the 
                actual target values (depends on rmse_type)
    '''
    assert model_type in ['decision', "randomforest",'adaboost', 'knn'], \
        "model_type can only be a value in ['decision', 'randomforest','adaboost', 'knn']"

    assert len(model_params) > 0, \
        "Empty input parameter list"
    
    try:
        assert not any(x.empty for x in [X_train, y_train, X_valid, y_valid]),\
            "Empty input data out of X_train, y_train, X_valid, and/or y_valid"
    except:
        pass

    model_data = pd.DataFrame()
    
    model_eval_dict = {'model_type':[], 
                       'model':[], 'train_score':[], 
                       'validation_score':[]}

    if run_test:
        try:
            assert not any(x.empty for x in [X_test, y_test]),\
                "Empty input data out of X_test, y_test"
        except:
            pass

    # helper function for validating classification accuracy
    def _calculate_rmse(model, model_dict : dict, valid_type : str, 
                        X_eval : pd.DataFrame, y_eval : pd.DataFrame) -> dict:
        ''' Parameters:
        ===============
        model - input model, can be of sklearn object type Lasso or Ridge
        model_dict - input dict of model param and evaluations
        valid_type - allows different rmse to be calculated, Allowed values: ['test', 'train', 'validation]
        X_eval - input data with selected features to predict with
        y_eval - data with target variable to evaluate predicted values against
        '''
        assert valid_type in ['test', 'train', 'validation'], \
        "valid_type can only be a value in ['test', 'train', 'validation']"
        
        if isinstance(X_eval, pd.DataFrame):
            y_pred = model.predict(X_eval.to_numpy())
            y_pred_prob = model.predict_proba(X_eval.to_numpy())
        else:
            y_pred = model.predict(X_eval)
            y_pred_prob = model.predict_proba(X_eval)
            
        if valid_type == "test":
            model_dict['{}_rmse'.format(valid_type)] = "TBD"
        else:
            accuracy = mse(y_true = y_eval.to_numpy(), y_pred = y_pred, 
                           squared = False)
            model_dict['{}_rmse'.format(valid_type)].append(accuracy)
        return({"rmse":model_dict, "predictions":y_pred, 'predict_prob':y_pred_prob})
    

    # define model from sklern by specified model_type
    if isinstance(X_train, pd.DataFrame):
        fit_model = model(**model_params).fit(X_train.to_numpy(), y_train.to_numpy())
    else:
        fit_model = model(**model_params).fit(X_train, y_train.to_numpy())

    # save model, <model_type>, and train, validation and optional test accurace scores to dict
    model_eval_dict['model'].append(fit_model)
    model_eval_dict['model_type'] = model_type
    train_pred = _calculate_rmse(fit_model, model_dict = model_eval_dict, 
                                 valid_type = "train", X_eval = X_train, y_eval = y_train)
    model_eval_dict = train_pred['score']
    
    # save validation accuracies
    valid_pred = _calculate_rmse(fit_model, model_dict = model_eval_dict, 
                                 valid_type = "validation", X_eval = X_valid, y_eval = y_valid)
    model_eval_dict = valid_pred['score']
    
    if run_test: # only calculate test rmse if specified
        test_pred = _calculate_rmse(fit_model, model_dict = model_eval_dict, 
                                 valid_type = "test", X_eval = X_test, y_eval = y_test)
        model_eval_dict = test_pred['score']
        model_eval_dict['pred'] = test_pred['predictions']
    
    # returns all predictions if selected
    if return_pred:
        return_dict = {"validate_pred":valid_pred['predictions'],
                       "train_pred":train_pred['predictions'],
                       "validate_pred_prob":valid_pred['predict_prob'],
                       "train_pred_prob":train_pred['predict_prob'],
                       "model_eval_dict":model_eval_dict}
        if run_test:
            return_dict["test_pred"] = test_pred['predictions']
            return_dict["test_pred_prob"] = test_pred['predict_prob']
        return(return_dict)
    else:
        return(pd.DataFrame(model_eval_dict))



def run(train : pd.Dataframe, valid : pd.Dataframe, model : str, 
        best_features : list):
    '''
    '''
    model_dict = {"decision":DecisionTreeRegressor, 
                 "randomforest":RandomForestRegressor, 
                 "linear":LinearRegression, 
                 "ada":AdaBoostRegressor}
    params = get_model_param("{}_params".format(model))
    cv = get_model_param("cross_validate_num")
    pred_var = get_model_param("target_var")
    best_params = find_best_model_params(model = model_dict[model](), 
                                         train_feat= train[best_features], 
                                         train_target = train[pred_var], 
                                         hyperparams = params, cv = cv)

    pred = train_and_evaluate_model(model = model_dict[model](),
                                    model_type = model,
                                    model_params = params, 
                                    X_train = train[best_features], y_train = train[pred_var], 
                                    X_valid = valid[best_features], y_valid = valid[pred_var],
                                    run_test = False,
                                    return_pred = True)
    return(pred)