##################################################################################
# Name of Script: remove_outliers.R
# Description: This code will detect outliers and deal with it in one of a few 
#               ways: 
#               - turn them into NAs (and let impute missing handle)
#               - capping outliers, for outside 1.5 * IQR limits, we use 
#                 5 percentile for lower limit and 95% percentile for upper limit
#               
# Arguments: N/A
# Output: 
# Contributors: rxu17
###################################################################################

#############################
# DEFINE PARAMETERS         #
#############################
# set working directories by operating system
rm(list=ls())
user <- Sys.info()[["user"]]

##################################
# DEFINE LIBRARIES AND FUNCTIONS #
##################################
pacman::p_load(data.table, tidyr)

detect_outliers <- function(input_df, dep_var, predictors = c()){
    # This method uses various outlier detection methods
    #
    # Parameters: 
    #   input_df: data.table
    #   dep_var: str, dependent variable of model
    #   predictors: list, list of predictor var to use in outlier model
    #
    # Returns: data.table with indicator of which row and col had the outlier

    # we predict on the dep_var using our list of predictors
    if(length(predictors) == 0) formula <- "{dep_var} ~ ." %>% glue
    else formula <- paste0("{dep_var} ~", paste0(predictors, collapse = " + ")) %>% glue
    mod <- lm(formula, data = input_df)
    cooksd <- cooks.distance(mod)

    # plot cook's distance
    plot(cooksd, pch = "*", cex = 2, main="Influential Obs by Cooks distance")  
    abline(h = 4 * mean(cooksd, na.rm = T), col="red")  # add cutoff line
    text(x = 1:length(cooksd)+1, y = cooksd, 
        labels=ifelse(cooksd > 4 * mean(cooksd, na.rm = T), names(cooksd),""), col = "red")  # add labels

    # threshold is 4* line
    input_df[, is_outlier := ifelse(cookds, 1, 0)]
    return(input_df)
}


treat_outliers <- function(input_df, out_method = "NA", cols_to_treat){
    # This method uses various outlier treatment methods to 
    # deal with the outliers detected from detect_outliers function
    #
    # Parameters:
    #   input_df: data.table, subset of original data with outliers
    #   out_method: str, ['NA', 'cap', 'removal'],
    #                NA - converts all rows to NA
    #                cap - takes 95% and 5% quntiles of our data, and 
    #                      caps our data at those values
    #                removal - removes those rows completely
    #   cols_to_treat: vector, cols that have outliers we want to handle
    # 
    # Returns: data.table of treated dataframe

    if(out_method == "NA"){
        input_df[, (cols_to_treat) := NA] 
    } else if (out_method == "cap"){
        # cap the data using caps at 5% and 95% quantiles of our data
        input_df[, .(lower_cap) = lapply(.SD, FUN = quantile(probs = 0.05)), 
                                                        .SDcols = cols_to_treat]
        input_df[, .(upper_cap) = lapply(.SD, FUN = quantile(probs = 0.95)), 
                                                        .SDcols = cols_to_treat]
        input_df[, lapply(.SD, function(x) ifelse(x > upper_cap, upper_cap, 
                                           ifelse(x < lower_cap, lower_cap, x))), 
                                           .SDcols = cols_to_treat]       
    } else if (out_method == "removal"){
        # completely remove rows of data
        input_df <- input_df[]
    }
    return(input_df)
}

