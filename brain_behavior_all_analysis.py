#!/usr/bin/env python3
"""
This code performs various data preprocessing and visualization tasks on brain and behavior data. Here is a summary of what the code does:

1. Load data from "dgrpool_brain_behavior_male.csv" and "dgrpool_brain_behavior_female.csv".
2. Drop unnecessary columns from the loaded data.
3. Check and drop columns with "Unnamed" in the name.
4. Check the type of each column and drop columns with string values.
5. Drop columns with "entropy" in the name.
6. Fill missing data with the median using nearest neighbor.
7. Apply z-score normalization to the data.
8. Drop NaN columns.
9. Perform MDS (Multidimensional Scaling) using sklearn on the behavior data.
10. Fit SVR (Support Vector Regression) on the brain data.
11. Merge the MDS output with the SVR output.
12. Plot the MDS output with plotly and color the points by morphology.
13. Save the MDS output as a CSV file.
14. Perform UMAP (Uniform Manifold Approximation and Projection) on the behavior data.
15. Plot the UMAP output with plotly.

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
entropy0_index = data_male.columns.get_loc("entropy0")
behav_male = data_male.iloc[:, :entropy0_index]
brain_male = data_male.iloc[:, entropy0_index:]

# Drop all columns after "DGRP" for data_female
entropy0_index = data_female.columns.get_loc("entropy0")
behav_female = data_female.iloc[:, :entropy0_index]
brain_female = data_female.iloc[:, entropy0_index:]

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
     
def svr_scores(df):
    scores = {}
    lsm = {}
    
    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Split the data into features (X) and target variable (y)
        X = df.drop(column, axis=1)
        y = df[column]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # Create and fit the SVR model
        svr = SVR()
        svr.fit(X_train, y_train)
        
        # Predict the target variable for the test set
        y_pred = svr.predict(X_test)
        
        # Calculate the R-squared score
        r2 = r2_score(y_test, y_pred)
        
        # Calculate the Least Squares Method (LSM) from the hyperplane
        mse = mean_squared_error(y_test, y_pred)
        lsm_score = np.sqrt(mse)
        scores_T = scores
        
        # Store the scores for the column
        scores = svr.predict(X)
    return scores

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

col_keep = ['Abs_volume(mm3)', "entropy0", "entropy1", "entropy2"]
data_morpho_male = brain_male[col_keep]
morpho_male = svr_scores(data_morpho_male)

data_morpho_female = brain_female[col_keep]
morpho_female = svr_scores(data_morpho_female)


# Merge behav_mds and morpho_male
behav_mds = pd.DataFrame(behav_mds)
behav_mds.columns = ['x', 'y']
behav_mds["morphology"] = morpho_male
behav_mds["dgrp"] = data_male["dgrp"].values

"""# Plot the mds output with plotly and color the points by morphology
fig = go.Figure()
fig.add_trace(go.Scatter(x=behav_mds['x'], y=behav_mds['y'], mode='markers', marker_color=behav_mds['morphology'], marker_size=10))
fig.update_layout(title='Behavioural Landscape from DGRP Males')
fig.show()"""

# Merge behav_female_mds and morpho_female
behav_female_mds = pd.DataFrame(behav_female_mds)
behav_female_mds.columns = ['x', 'y']
behav_female_mds["morphology"] = morpho_female
behav_female_mds["dgrp"] = data_female["dgrp"].values

"""# Plot the mds output with plotly and color the points by morphology
fig = go.Figure()
fig.add_trace(go.Scatter(x=behav_female_mds['x'], y=behav_female_mds['y'], mode='markers', marker_color=behav_female_mds['morphology'], marker_size=10))
fig.update_layout(title='Behavioural Landscape from DGRP Females')
fig.show()"""

"""# Save behav_mds and behav_female_mds as csv
behav_mds.to_csv("behav_male_mds.csv", index=False)
behav_female_mds.to_csv("behav_female_mds.csv", index=False)"""

# UMAP on behav_male and behav_female

reducer = umap.UMAP()
umap_male = reducer.fit_transform(behav_male)
umap_female = reducer.fit_transform(behav_female)
 
"""# Plot the umap output with plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=umap_male[:, 0], y=umap_male[:, 1], mode='markers', marker_color=data_male["DGRP"]))
fig.update_layout(title='Behavioural Landscape from DGRP Males')
fig.show()

fig.add_trace(go.Scatter(x=umap_female[:, 0], y=umap_female[:, 1], mode='markers', marker_color=data_female["DGRP"]))
fig.update_layout(title='Behavioural Landscape from DGRP Females')
fig.show()

"""
#Merge umap_male and morpho_male
umap_male = pd.DataFrame(data=umap_male)
umap_male.columns = ['x', 'y'] 
umap_male["morphology"] = morpho_male
umap_male["dgrp"] = data_male["dgrp"].values
umap_male["entropy0"] = data_male["entropy0"].values
umap_male["entropy1"] = data_male["entropy1"].values
umap_male["entropy2"] = data_male["entropy2"].values

# Merge umap_female and morpho_female
umap_female = pd.DataFrame(data=umap_female)
umap_female.columns = ['x', 'y']
umap_female["morphology"] = morpho_female
umap_female["dgrp"] = data_female["dgrp"].values
umap_female["entropy0"] = data_female["entropy0"].values
umap_female["entropy1"] = data_female["entropy1"].values
umap_female["entropy2"] = data_female["entropy2"].values


"""# plot the umap output with plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=umap_male['x'], y=umap_male['y'], mode='markers', marker_color=umap_male['morphology'], marker_size=10))
fig.update_layout(title='Behavioural Landscape from DGRP Males')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=umap_female['x'], y=umap_female['y'], mode='markers', marker_color=umap_female['morphology'], marker_size=10))
fig.update_layout(title='Behavioural Landscape from DGRP Females')
fig.show()

"""
# Save umap_male and umap_female as csv
umap_male.to_csv("umap_male.csv", index=False)
umap_female.to_csv("umap_female.csv", index=False)

