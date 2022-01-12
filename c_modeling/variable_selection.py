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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA


def univariate_selection(input_df, x_cols, y_col, num_features):
    ''' This method analyzes the statistical importance (chi2) of each variable
        and picks highest performing ones

        Parameters:
            input_df: dataframe input
            x_cols: list, names of predictor vars
            y_col: str, name of dependent var
            num_features: int, [1, number of predictors], the number of features
            you want bested for your model
        
        Returns: dataframe with top best (num_features) features
    '''
    assert(num_features < len(input_df.columns)), \
            "Error: number of features exceed number of variables in input_df"
    assert(set(x_cols) < set(input_df.columns)), \
            "Error: Certain predictors not present in input_df"
    assert(y_col in input_df.columns), \
            "Error: dependent var not present in input_df"

    X = input_df[x_cols]
    y = input_df[y_col]  #target column i.e survival
    # use SelectKBest to extract top 10 best features
    best = SelectKBest(score_func = chi2, k = num_features)
    fit = best.fit(X,y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization 
    featureScores = pd.concat([df_columns,df_scores], axis = 1)
    featureScores.columns = ['Variables','Score'] 
    print(featureScores.nlargest(num_features,'Score'))  #print 10 best features
    return(input_df[featureScores['Variables'].unique()])


def pca(input_df, x_cols, y_col, num_features):
    ''' NOTE: PCA methods need scaling done beforehand
        This method analyzes the PCA (Principal Component Analysis) of each
        feature in a model and decides which ones are the most import.

        Parameters:
            input_df: dataframe input
            x_cols: list, names of predictor vars
            y_col: str, name of dependent var
            num_features: int, [1, number of predictors], the number of features
            you want bested for your model
        
        Returns: pca object with ideal number of components to keep
    '''
    X = input_df[x_cols]
    pca = PCA(n_components = num_features)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    return(pca)

def step_forward_selection():
    pass

def rfe():
    pass

def feature_importance():
    pass

def corr_importance(input_df, x_cols, y_col, threshold):
    ''' This method gets the correlation of predictor vars with
        target vars and decides which one matches threshold level

        Parameters:
            input_df: dataframe input
            x_cols: list, names of predictor vars
            y_col: str, name of dependent var
            threshold: float, [0...1] level at which correlation
                        with target var should be above
        
        Returns: df with cols that match threshold level
    '''
    corr_df = input_df[input_df.columns[1:]].corr()[y_col][:]
    corr_df = corr_df[corr_df['corr'] >= threshold]
    return(corr_df)


def main():
    input_df = pd.read_csv("{}/training_v2.csv".format(os.getcwd()))
    feature_df.to_csv("{}/feature_select_df.csv".format(os.getcwd()))


if __name__ == "__main__":
    main()