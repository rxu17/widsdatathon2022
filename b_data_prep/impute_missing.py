'''
Description: This script searches for missingness in a dataset and based on
               user selection, takes an imputation method and fills in the missingness

Arguments: None
How To Use: import 
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


def run_knn(input : pd.DataFrame, impute_var):
    '''
    '''


def run(input : pd.DataFrame, impute_vars : list = None, imputer : list = None) -> pd.DataFrame:
    ''' Select imputation method
    '''
    if impute_vars is None and imputer is None:
        return(input)
    impute_dict = {impute_vars[i]:imputer[i] for i in len(impute_vars)}
    for impute_var, imputer in impute_dict.items():
        if imputer == "knn":
            updated = run_knn(input, impute_var)
        elif imputer == "wgt_avg":
            updated = []
        elif imputer == "mode":
            mode = input[impute_var].mode()
            updated = input.replace(to_replace = NA, value = mode)
        elif imputer == "hot_deck":
            updated = run_hot_deck(impute_var)
        elif imputer == "linear_reg":
            updated = run_linear_reg(impute_var)
        elif imputer == "remove_thres":
            updated = run_remove_threshold(impute_var)
    return(updated)


check_threshold <- function(input_df, th_pct = 5){
    # This method checks the percentage of missing
    # and drops variables that are greater than that threshold
    #
    # Parameters:
    #   input_df: dataframe with missing values
    #   th_pct: int, {0...100} percentage of missing val
    #           in dataset that you would like to be cutoff pt
    #
    # Returns: data.table with not meeting threshold 
    # variables removed
    #
    assert_that(th_pct <= 1 & th_pct >= 0, 
                msg = "threshold precentage is not in the range of 0...1")
    p_miss <- function(x){  # helper function checks for pctage missing in a col
                    (sum(is.na(x))/length(x))*100
                    }
    p_miss_mat <- apply(input_df, 2, p_miss) # applies check across all cols
    p_miss_mat$id_col <- 1
    setDT(p_miss_mat)
    # reshapes data, from variables being wide to long
    p_miss_mat <- melt(p_miss_mat, id.vars = "id_col")

    # find vars that don't meet threshold and drop
    var_to_drop <- p_miss_mat[value > th_pct]$var %>% unique
    missing_removed <- input_df[, !(var_to_drop), with = F]
    return(missing_removed)
}


impute_method_selection <- function(method = "knn", input_df){
    # This method takes in a dataset with missing values and depending on
    # imputation method selected, returned imputed dataset
    #
    # Parameters:
    #   method: [knn, random_forest, hot_deck, linear, em]
    #   input_df: dataframe with missing values
    #
    # Returns: imputed dataset
    #
    allowed_met <- c('knn', 'random_forest', 'mice','hot_deck', 'linear', 'em')
    assert_that(method %in% allowed_met, 
            msg = glue("You must pick a method from available methods: {allowed_met}"))
    if (method == "knn"){
        # weighted knn (K-nearest neigbors)
        input_mat <- as.matrix(input_df)

        # cross validate for best lambda
        cv_best <- cv.wNNSel(x, kernel = "gaussian", x.dist = "euclidean", 
                            method = "2", m.values = seq(2, 8, by = 2),
                            lambda.values = seq(0, 0.6, by = 0.01)[-1], times.max = 5)

        imputed_df <- wNNSel.impute(x, k, useAll = TRUE, 
                                    x.initial = NULL, x.dist = "euclidean",
                                    kernel = "gaussian", lambda = cv_best$lambda.opt, 
                                    convex = TRUE,
                                    method = "2", m = cv_best$m.opt, c = 0.3,
                                    verbose = TRUE, verbose2 = FALSE)
    } else if (method == "random_forest"){
        # uses random forest method to predict missingness
        input_mat <- as.matrix(input_df)
        imputed_df <- missForest(input_mat, maxiter = 10, ntree = 100, variablewise = FALSE,
                    decreasing = FALSE, verbose = FALSE,
                    mtry = floor(sqrt(ncol(xmis))), replace = TRUE,
                    classwt = NULL, cutoff = NULL, strata = NULL,
                    sampsize = NULL, nodesize = NULL, maxnodes = NULL,
                    xtrue = NA, parallelize = c('no', 'variables', 'forests'))

    } else if (method == "hot_deck"){
        # Method where missing value receives valid value from a case randomly 
        # chosen from those cases which are maximally similar to the missing one, 
        # based on some background variables specified by the user
        imputed_df <- hotdeck(data, variable = c(), ord_var = c(),
                             impNA = TRUE, imp_var = TRUE,
                            imp_suffix = "imp")

    } else if (method == "linear"){
        # uses linear regression to predict missing
        imputed_df <- waverr(RawData = input_mat, Nrepeats = 5)
    } else if (method == "em"){
        # uses Expectation Maximization method
        imputed_df <- em.mix(s, start, prior=1, maxits=1000, showits=TRUE, eps=0.0001)
    } else if (method == "mice"){
        # uses Multivariate Imputation by Chained Equations method
        imputed_df <- mice(input_df, m = 5, method = "rf")
        imputed_df <- complete(imputed_df)
    }
    return(imputed_df)
}