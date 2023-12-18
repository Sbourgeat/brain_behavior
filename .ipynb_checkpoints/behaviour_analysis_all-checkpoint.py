# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr

def split_string_with_dgrp(df, is_sex = True):
    """
    This function takes a dataframe as input and splits the 'genotype' column into two separate lists: DGRP and sex.
    
    Parameters:
    df (DataFrame): The input dataframe containing the 'genotype' and 'sex' columns.
    
    Returns:
    DGRP (list): A list of DGRP values extracted from the 'genotype' column.
    sex (list): A list of sex values extracted from the 'sex' column.
    """
    DGRP=[]
    sex = []
    for i in range(len(df)):
        if 'dgrp' in df.iloc[i, 1]:
            DGRP.append(df.iloc[i, 1])
            if is_sex:
                s= df.iloc[i,2]
                sex.append(s)
    return DGRP,sex

def merge_data(brain, behav, DGRP, sex=None, is_b2 = False):
    """
    This function merges the brain and behavior data based on the DGRP and sex values.
    
    Parameters:
    brain (DataFrame): The brain data dataframe.
    behav (DataFrame): The behavior data dataframe.
    DGRP (list): A list of DGRP values.
    sex (list): A list of sex values.
    
    Returns:
    merged_df (DataFrame): The merged dataframe containing the brain and behavior data.
    """
    if is_b2:
        data = behav[behav['genotype'].isin(DGRP)]
    else:
        data = behav[behav['genotype'].isin(DGRP) & behav["head_scanned"]==True]
    data['genotype'] = data['genotype'].apply(lambda x: 'DGRP_0' + x.split('dgrp')[1] if len(x.split('dgrp')[1]) == 2 else 'DGRP_' + x.split('dgrp')[1])
    brain['DGRP'] = brain['DGRP'].apply(lambda x: 'DGRP_0' + x if len(x) == 2 else 'DGRP_'+ x)

    
    data.rename(columns={'genotype': 'DGRP'}, inplace=True)


    if sex == None:
        merged_df = pd.merge(brain, data, on='DGRP')
    else:
        merged_df = pd.merge(brain, data, on=['DGRP', 'sex'])
    
    return merged_df

def calculate_pvalues(df):
    """
    This function calculates the p-values for the correlation matrix of a dataframe.
    
    Parameters:
    df (DataFrame): The input dataframe.
    
    Returns:
    pvalues (DataFrame): The dataframe containing the p-values for the correlation matrix.
    """
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
    return pvalues

behav = pd.read_csv("/Users/skumar/Documents/PhD/BrainAnalysis/Behavior/summary.csv")
brain = pd.read_csv("/Users/skumar/Project/PHD_work/GWAS/dataset/vol_hratio.csv", sep=",")
behav_2 = pd.read_csv("/Users/skumar/Documents/PhD/BrainAnalysis/Behavior/brain_behavior/pair_compare_sd.csv")

DGRP,sex = split_string_with_dgrp(behav)
merged_df = merge_data(brain, behav, DGRP, sex)


DGRP, sex = split_string_with_dgrp(behav_2, is_sex=False)

merged_df = merge_data(merged_df, behav_2, DGRP, is_b2=True)




correlation_matrix = merged_df[["abs_volume","h_ratio","activity","correct_choices","SE"]].corr()
"""
fig = px.imshow(correlation_matrix)
fig.show()

fig = px.imshow(calculate_pvalues(merged_df[["abs_volume","h_ratio","activity","correct_choices","frac_time_on_shocked"]]))
fig.show()
"""

p_values = calculate_pvalues(merged_df[["abs_volume", "h_ratio", "activity", "correct_choices", "SE"]])

fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=["Absolute volume (std)", "H-ratio", "Activity", "Correct number of choices", "Place learning ability"],
    y=["Absolute volume (std)", "H-ratio", "Activity", "Correct number of choices", "Place learning ability"],
    colorscale="Blues",
    colorbar=dict(title="Correlation Coefficient")
))

names = ["Absolute volume (std)", "H-ratio", "Activity", "Correct number of choices", "Place learning ability"]
annotations = []
for i, row in enumerate(correlation_matrix.values):
    for j, value in enumerate(row):
        annotations.append(
            dict(
                x=names[j],
                y=names[i],
                text=f"p-value: {p_values.iloc[i, j]:.3f}",
                showarrow=False,
                font=dict(color="white" if abs(value) > 0.5 else "black")
            )
        )

fig.update_layout(
    title="Correlation Coefficient and p-values",
    annotations=annotations,
    xaxis=dict(title="Variable"),
    yaxis=dict(title="Variable"),
)

fig.show()