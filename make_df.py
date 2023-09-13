import pandas as pd

def select_columns(df, col_names):
    """
    Takes a pandas df and column names and return a new df with only the specified columns.
    """
    df_new = {}
    for idx, names in enumerate(col_names):
        df_new[str(names)]=df[names].values
    df_new = pd.DataFrame(df_new)
    return df_new


def is_dgrp_in_file(df1, df2):
    colname2 = df2.columns
    colname = df1.columns
    for idx, dgrp in enumerate(df2[colname2[0]]):
        if dgrp not in df1[colname[0]].values:
            df2 = df2.drop(index=idx)
    return df2


def only_dgrp_data(df1, df2):
    """
    Takes two df and extract only one with similar dgrp line
    """
    colnames1 = df1.columns
    colnames2 = df2.columns 
    new_df = []
    colnames =[colnames1[0], colnames1[1], colnames2[1], colnames2[2]]
    for idx2, name2 in enumerate(df2[colnames2[0]]):
        for idx1, name1 in enumerate(df1[colnames1[0]]):
           
            if name1 == name2 :
                new_df.append([name1, df1[colnames1[1]][idx1], df2[colnames2[1]][idx2], df2[colnames2[2]][idx2]])
                
    
    new_df = pd.DataFrame(new_df)
    new_df.columns = colnames
    return new_df


def merge_df(df1, df2, col_names1, col_names2, diet_type, entropy ):
    """
    Takes two df and two lists of column names and return a merged df from the input dsf
    """
    df1 = select_columns(df1,[col_names1[0], col_names1[entropy]])

    #df2 = is_dgrp_in_file(df1,df2)
    #choose diet type 
        
    df2 = select_columns(df2, col_names2)
    df_merged = only_dgrp_data(df1, df2)

    df_merged = df_merged[df_merged['diet']==diet_type]

   # print(df2)

       
   # df2 = select_columns(df2, col_names2)
    
    
    #df_merged = df1.append(df2)
    return df_merged



# Create a function that combines the brain dataframe witht the metabolism dataframe and 
# make a correlation matrix between all three entferent metabolites

def merge_all_param(df1, df2, diet_type):
    col_names1 = df1.columns
    df1_E0 = select_columns(df1, [col_names1[0], col_names1[2], col_names1[3]])
    df1_E1 = select_columns(df1, [col_names1[0],col_names1[2], col_names1[4]])
    df1_E2 = select_columns(df1, [col_names1[0], col_names1[2],col_names1[5]])

    # a function to merge df 1 and df2
    
    df_merge_ent0 = only_dgrp_data_all_meta(df1_E0, df2)
    df_merge_ent1 = only_dgrp_data_all_meta(df1_E1, df2)
    df_merge_ent2 = only_dgrp_data_all_meta(df1_E2, df2)


    return df_merge_ent0, df_merge_ent1, df_merge_ent2






def only_dgrp_data_all_meta(df1, df2):
    """
    Takes two df and extract only one with similar dgrp line with all metabolites 
    """
    # Assuming the first column is the common column for merging
    left_on_column = df1.columns[0]
    right_on_column = df2.columns[0]

    # Merge the dataframes based on the specified columns
    merged_df = df1.merge(df2, left_on=left_on_column, right_on=right_on_column, how='inner')
    column_to_drop = 'DGRP_line'
    merged_df = merged_df.drop(columns=column_to_drop)
    

   
    return merged_df


