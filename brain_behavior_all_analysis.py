
import pandas as pd


# load data from brain_behavior/dgrpool_brain_behavior_female.csv and brain_behavior/dgrpool_brain_behavior_male.csv
data_male = pd.read_csv("dgrpool_brain_behavior_male.csv")
data_female = pd.read_csv("dgrpool_brain_behavior_female.csv")


# from the data_male drop all the column after "DGRP"

# Get the index of the column named "DGRP"
dgrp_index = data_male.columns.get_loc("entropy0")
# Select all columns before the "DGRP" column
behav_male = data_male.iloc[:, :dgrp_index]
# Select all columns starting from the "DGRP" column
brain_male = data_male.iloc[:, dgrp_index:]


#same for female
dgrp_index = data_female.columns.get_loc("entropy0")
behav_female = data_female.iloc[:, :dgrp_index]
brain_female = data_female.iloc[:, dgrp_index:]


# check the df and drop if the colname has Unnamed 
def check_unnamed(df):
    for col in df.columns:
        if 'Unnamed' in col:
            df.drop(col, axis=1, inplace=True)
    return df


# Function that looks at the type of each columns and drop the column if there is a str
def check_type(df):
    for col in df.columns:
        if type(df[col][0]) == str:
            df.drop(col, axis=1, inplace=True)
    return df


#Function that drops columns having entropy in the name
def drop_entropy(df):
    for col in df.columns:
        if 'entropy' in col:
            df.drop(col, axis=1, inplace=True)
    return df


behav_male = drop_entropy(check_type(check_unnamed(behav_male)))
behav_female = drop_entropy(check_type(check_unnamed(behav_female)))





#  for behav_male and behave_female, use nearest neighbour to fill missing data
behav_male = behav_male.fillna(behav_male.median())
behav_female = behav_female.fillna(behav_female.median())


#Apply a zscore to each columns
behav_male = (behav_male - behav_male.mean()) / behav_male.std()
behav_female = (behav_female - behav_female.mean()) / behav_female.std()

# drop NaN columns
behav_male = behav_male.dropna(axis=1)
behav_female = behav_female.dropna(axis=1)




# perform a MDS using sklearn on behav_male 
from sklearn.manifold import MDS
mds = MDS(n_components=2)
behav_mds = mds.fit_transform(behav_male)
behav_female_mds = mds.fit_transform(behav_female)





# on brain_male and brain_female fit a SVR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np



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



col_keep = ['Abs_volume(mm3)', "entropy0", "entropy1", "entropy2"]
data_morpho_male = brain_male[col_keep]
morpho_male=svr_scores(data_morpho_male)


# Same for female
data_morpho_female = brain_female[col_keep]
morpho_female=svr_scores(data_morpho_female)




# merge behav_mds and morpho_male 
behav_mds = pd.DataFrame(behav_mds)
behav_mds.columns = ['x', 'y']
behav_mds["morphology"] = morpho_male
behav_mds["dgrp"] = data_male["dgrp"].values

# plot the mds output with plotly and the color the points by morphology 
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=behav_mds['x'], y=behav_mds['y'], mode='markers', marker_color=behav_mds['morphology'], marker_size=10))
fig.update_layout(title='Behavioural Landscape from DGRP Males')
fig.show()

#same for behave_female_mds
behav_female_mds = pd.DataFrame(behav_female_mds)
behav_female_mds.columns = ['x', 'y']
behav_female_mds["morphology"] = morpho_female
behav_female_mds["dgrp"] = data_female["dgrp"].values

# plot the mds output with plotly and the color the points by morphology
fig = go.Figure()
fig.add_trace(go.Scatter(x=behav_female_mds['x'], y=behav_female_mds['y'], mode='markers', marker_color=behav_female_mds['morphology'], marker_size=10))
fig.update_layout(title='Behavioural Landscape from DGRP females')
fig.show()


# save as csv behav_mds and behav_female_mds
behav_mds.to_csv("behav_male_mds.csv")
behav_female_mds.to_csv("behav_female_mds.csv")