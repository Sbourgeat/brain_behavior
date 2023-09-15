#!/usr/bin/env python3
"""


"""
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import umap

# Load data from brain_behavior/dgrpool_brain_behavior_female.csv and brain_behavior/dgrpool_brain_behavior_male.csv
data_male = pd.read_csv("dgrpool_brain_behavior_male.csv")
data_female = pd.read_csv("dgrpool_brain_behavior_female.csv")

# Drop all columns after "DGRP" for data_male
dgrp_index = data_male.columns.get_loc("DGRP")
behav_male = data_male.iloc[:, :dgrp_index]
brain_male = data_male.iloc[:, dgrp_index:]

# Drop all columns after "DGRP" for data_female
dgrp_index = data_female.columns.get_loc("DGRP")
behav_female = data_female.iloc[:, :dgrp_index]
brain_female = data_female.iloc[:, dgrp_index:]

# Check and drop columns with "Unnamed" in the name
def check_unnamed(df):
    for col in df.columns:
        if 'Unnamed' in col:
            df.drop(col, axis=1, inplace=True)
    return df

# Check the type of each column and drop columns with string values
def check_type(df):
    for col in df.columns:
        if type(df[col][0]) == str:
            df.drop(col, axis=1, inplace=True)
    return df

# Drop columns with "entropy" in the name
def drop_entropy(df):
    for col in df.columns:
        if 'entropy' in col:
            df.drop(col, axis=1, inplace=True)
    return df

# Apply check_unnamed, check_type, and drop_entropy to behav_male and behav_female
behav_male = drop_entropy(check_type(check_unnamed(behav_male)))
behav_female = drop_entropy(check_type(check_unnamed(behav_female)))

# Fill missing data with median using nearest neighbor for behav_male and behav_female
behav_male = behav_male.fillna(behav_male.median())
behav_female = behav_female.fillna(behav_female.median())

# Apply z-score normalization to behav_male and behav_female
behav_male = (behav_male - behav_male.mean()) / behav_male.std()
behav_female = (behav_female - behav_female.mean()) / behav_female.std()

# Drop NaN columns for behav_male and behav_female
behav_male = behav_male.dropna(axis=1)
behav_female = behav_female.dropna(axis=1)

# Perform MDS using sklearn on behav_male
mds = MDS(n_components=2)
behav_mds = mds.fit_transform(behav_male)
behav_female_mds = mds.fit_transform(behav_female)

# Fit SVR on brain_male and brain_female
def svr_scores(df):
    scores = []
    lsm = []
    
    for column in df.columns:
        X = df.drop(column, axis=1)
        y = df[column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        svr = SVR()
        svr.fit(X_train, y_train)
        
        y_pred = svr.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        lsm_score = np.sqrt(mse)
        
        scores.append(svr.predict(X))
        lsm.append(lsm_score)
        
    return scores

col_keep = ['Abs_volume(mm3)', "entropy0", "entropy1", "entropy2"]
data_morpho_male = brain_male[col_keep]
morpho_male = svr_scores(data_morpho_male)

data_morpho_female = brain_female[col_keep]
morpho_female = svr_scores(data_morpho_female)

# Merge behav_mds and morpho_male
behav_mds = pd.DataFrame(behav_mds)
behav_mds.columns = ['x', 'y']
behav_mds["morphology"] = morpho_male
behav_mds["dgrp"] = data_male["DGRP"].values

# Plot the mds output with plotly and color the points by morphology
fig = go.Figure()
fig.add_trace(go.Scatter(x=behav_mds['x'], y=behav_mds['y'], mode='markers', marker_color=behav_mds['morphology'], marker_size=10))
fig.update_layout(title='Behavioural Landscape from DGRP Males')
fig.show()

# Merge behav_female_mds and morpho_female
behav_female_mds = pd.DataFrame(behav_female_mds)
behav_female_mds.columns = ['x', 'y']
behav_female_mds["morphology"] = morpho_female
behav_female_mds["dgrp"] = data_female["DGRP"].values

# Plot the mds output with plotly and color the points by morphology
fig = go.Figure()
fig.add_trace(go.Scatter(x=behav_female_mds['x'], y=behav_female_mds['y'], mode='markers', marker_color=behav_female_mds['morphology'], marker_size=10))
fig.update_layout(title='Behavioural Landscape from DGRP Females')
fig.show()

# Save behav_mds and behav_female_mds as csv
behav_mds.to_csv("behav_male_mds.csv", index=False)
behav_female_mds.to_csv("behav_female_mds.csv", index=False)

# UMAP on behav_male and behav_female
reducer = umap.UMAP()
umap_male = reducer.fit_transform(behav_male)
umap_female = reducer.fit_transform(behav_female)
 
# Plot the umap output with plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=umap_male[:, 0], y=umap_male[:, 1], mode='markers', marker_color=data_male["DGRP"]))
fig.update_layout(title='Behavioural Landscape from DGRP Males')
fig.show()

fig.add_trace(go.Scatter(x=umap_female[:, 0], y=umap_female[:, 1], mode='markers', marker_color=data_female["DGRP"]))
fig.update_layout(title='Behavioural Landscape from DGRP Females')
fig.show()
