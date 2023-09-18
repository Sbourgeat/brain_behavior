# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pandas as pd

# import data from data_male.csv and data_female.csv
def import_data():
    """
    Reads the data from the 'data_male.csv' and 'data_female.csv' files and returns them as separate dataframes.

    Returns:
        data_male (pandas.DataFrame): The dataframe containing the data from the 'data_male.csv' file.
        data_female (pandas.DataFrame): The dataframe containing the data from the 'data_female.csv' file.
    """
    data_male = pd.read_csv("data_male.csv")
    data_female = pd.read_csv("data_female.csv")
    return data_male, data_female


# import volume and entropy data from vol_entropy.csv
def import_vol_entropy():
    """
    Reads the 'entropy_vol_sep2023.csv' file and returns the contents as a pandas DataFrame.

    Returns:
        pandas.DataFrame: The contents of the 'entropy_vol_sep2023.csv' file.
    """
    vol_entropy = pd.read_csv("entropy_vol_sep2023.csv")
    return vol_entropy

# change the column name DGRP of vol_entropy to dgrp and for each row in dgrp, add DGRP_ if len(item)==3 else add DGRP_0
def normalize_colname():
    """
    Normalize the column names in the vol_entropy dataframe.

    Returns:
        pandas.DataFrame: The vol_entropy dataframe with normalized column names.
    """
    vol_entropy['dgrp'] = vol_entropy['DGRP'].apply(lambda x: 'DGRP_0' + str(x) if len(str(x)) == 2 else 'DGRP_'+ str(x))
    return vol_entropy

# separate vol_entropy by Sex == male and Sex == female
def separate_vol_entropy():
    """
    Separate the given 'vol_entropy' DataFrame into two separate DataFrames: one for male and one for female.
    
    Returns:
        male (DataFrame): The DataFrame containing only the male rows from the 'vol_entropy' DataFrame.
        female (DataFrame): The DataFrame containing only the female rows from the 'vol_entropy' DataFrame.
    """
    male = vol_entropy[vol_entropy['Sex'] == 'male']
    female = vol_entropy[vol_entropy['Sex'] == 'female']
    return male, female

# merge the data_male and male dataframe by dgrp 
def merge_data_male(df1,df2):
    """
    Merge the dataframes `df1` and `df2` on the column 'dgrp'.
    
    Args:
        df1 (DataFrame): The first dataframe to be merged.
        df2 (DataFrame): The second dataframe to be merged.
        
    Returns:
        DataFrame: The merged dataframe.
    """
    male = pd.merge(df1, df2, on='dgrp')
    return male

# merge the data_female and female dataframe by dgrp
def merge_data_female(df1,df2):
    """
    Merge two dataframes based on the 'dgrp' column and return the resulting dataframe.
    
    Parameters:
        df1 (pandas.DataFrame): The first dataframe to be merged.
        df2 (pandas.DataFrame): The second dataframe to be merged.
        
    Returns:
        female (pandas.DataFrame): The merged dataframe.
    """
    female = pd.merge(df1, df2, on='dgrp')
    return female


# apply the functions to create the merged df




data_male, data_female = import_data()
vol_entropy = import_vol_entropy()
vol_entropy = normalize_colname()
male, female = separate_vol_entropy()
merged_data_male = merge_data_male(data_male, male)
merged_data_female = merge_data_female(data_female, female)


# write to csv as dgrpool_brain_behavior_male.csv and dgrpool_brain_behavior_female.csv
merged_data_male.to_csv('dgrpool_brain_behavior_male.csv')
merged_data_female.to_csv('dgrpool_brain_behavior_female.csv')

