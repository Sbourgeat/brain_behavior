"""
Here, we will compare the KL divergence between each genotypes scored with 
the bottleneck distances of the same genotypes.


The correlation between KL divergence and bottleneck distances can provide insights into the relationship between these two measures.

KL divergence is a measure of how one probability distribution diverges from a second, expected probability distribution. 
In the context of your code, it seems to be used to measure the divergence between two genetic populations (males and females).

Bottleneck distances, on the other hand, are a measure of the difference between two persistence diagrams, which are tools used in topological data analysis to study the shape of data. 
In your context, it seems to be used to measure the genetic distance between two populations.

If there is a strong positive correlation between KL divergence and bottleneck distances, 
it suggests that as the KL divergence between two populations increases (i.e., as the genetic divergence between the populations increases), 
the bottleneck distance also increases. This could imply that the two measures are capturing similar aspects of genetic divergence.

On the other hand, if there is a weak or no correlation, it suggests that the two measures are capturing different aspects of genetic divergence. 
This could be useful information for researchers studying these populations, as it could suggest that they need to consider both measures to fully understand the genetic 
divergence between the populations.

Please note that this is a general interpretation based on the typical uses of KL divergence and bottleneck distances. 
The specific significance of the correlation in your study would depend on the specifics of your data and research question.



"""

import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import numpy as np 

# Define a function to transform the string
def transform_string(s):
    if isinstance(s, str) and s.startswith('dgrp'):
        return 'DGRP_' + s[4:].upper()
    return s


# import the KL divergence
kl = pd.read_csv('kl_matrix_new_working.csv')


# Apply the function to the column names
kl.columns = kl.columns.map(transform_string)
kl.set_index(kl.columns[0], inplace=True)
# Apply the function to the row index names
kl.index = kl.index.map(transform_string)

# import the bottleneck distances whic is a np file
bottleneck = pd.read_csv('/home/samuel/brainMorpho/Analysis_results/homology2voxel_pairwise/bottleneck_matrix.csv')
bottleneck.set_index(bottleneck.columns[0], inplace=True)


# Split the DataFrame into two based on the column names
bottleneck_male = bottleneck.filter(regex='_male')
bottleneck_female = bottleneck.filter(regex='_female')


# Further split the DataFrames based on the row names
bottleneck_male = bottleneck_male.loc[bottleneck_male.index.str.contains('_male')]
bottleneck_female = bottleneck_female.loc[bottleneck_female.index.str.contains('_female')]

 # delete column index 


# Define a function to transform the string
def transform_string(s):
    if isinstance(s, str):
        return 'DGRP_' + s.split('_')[0].upper()
    return s

# Apply the function to the column names
bottleneck_male.columns = bottleneck_male.columns.map(transform_string)
bottleneck_female.columns = bottleneck_female.columns.map(transform_string)
# Apply the function to the row index names
bottleneck_male.index = bottleneck_male.index.map(transform_string)
bottleneck_female.index = bottleneck_female.index.map(transform_string)

# filter both dataframes to keep only the rows sharing the same DGRP_lines

# Get the common columns
common_columns = kl.columns.intersection(bottleneck_male.columns)

# Filter the DataFrames to keep only the common columns
kl_male = kl.loc[common_columns, common_columns]
bottleneck_male = bottleneck_male.loc[common_columns, common_columns]

# Get the common columns
common_columns = kl.columns.intersection(bottleneck_female.columns)

# Filter the DataFrames to keep only the common columns
kl_female = kl.loc[common_columns, common_columns]
bottleneck_female = bottleneck_female.loc[common_columns, common_columns]




# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler and transform the data
kl_male_normalized = scaler.fit_transform(kl_male.values.flatten().reshape(-1, 1))
bottleneck_male_normalized = scaler.fit_transform(bottleneck_male.values.flatten().reshape(-1, 1))

# Fit the scaler and transform the data
kl_female_normalized = scaler.fit_transform(kl_female.values.flatten().reshape(-1, 1))
bottleneck_female_normalized = scaler.fit_transform(bottleneck_female.values.flatten().reshape(-1, 1))
# Create a new DataFrame for the male plot

df_plot_male = pd.DataFrame({
    'KL Male': kl_male_normalized.flatten(),
    'Bottleneck Male': bottleneck_male_normalized.flatten()
})

# Create a new DataFrame for the female plot
df_plot_female = pd.DataFrame({
    'KL Female': kl_female_normalized.flatten(),
    'Bottleneck Female': bottleneck_female_normalized.flatten()
})

# Create the scatter plot
fig = go.Figure()

# Add male scatter trace
fig.add_trace(go.Scatter(
    x=df_plot_male['KL Male'],
    y=df_plot_male['Bottleneck Male'],
    mode='markers',
    name='Males',
    marker=dict(color='red')
))

# Add female scatter trace
fig.add_trace(go.Scatter(
    x=df_plot_female['KL Female'],
    y=df_plot_female['Bottleneck Female'],
    mode='markers',
    name='Females',
    marker=dict(color='purple')
))

# Add labels and title
fig.update_layout(
    title='Comparison between Normalized KL and Normalized Bottleneck',
    xaxis_title='Normalized KL',
    yaxis_title='Normalized Bottleneck',
     plot_bgcolor='rgba(0,0,0,0)',  # this sets the plot background color to transparent
    paper_bgcolor='rgba(0,0,0,0)',  # this sets the paper (around the plot) background color to transparent

)

# Show the plot
fig.show()

 #save as html
fig.write_html("KL_mcmc_vs_bottleneck.html")



# Calculate the correlation coefficient for males
correlation_male = np.corrcoef(df_plot_male['KL Male'], df_plot_male['Bottleneck Male'])[0, 1]

# Calculate the correlation coefficient for females
correlation_female = np.corrcoef(df_plot_female['KL Female'], df_plot_female['Bottleneck Female'])[0, 1]

print(f"Correlation coefficient for males: {correlation_male}")
print(f"Correlation coefficient for females: {correlation_female}")







from sklearn.metrics import r2_score

# Fit a line to the male data
male_line = np.polyfit(df_plot_male['KL Male'], df_plot_male['Bottleneck Male'], 1)
male_r2 = r2_score(df_plot_male['Bottleneck Male'], np.polyval(male_line, df_plot_male['KL Male']))

# Fit a line to the female data
female_line = np.polyfit(df_plot_female['KL Female'], df_plot_female['Bottleneck Female'], 1)
female_r2 = r2_score(df_plot_female['Bottleneck Female'], np.polyval(female_line, df_plot_female['KL Female']))

# Create the scatter plot
fig = go.Figure()

# Add male scatter trace
fig.add_trace(go.Scatter(
    x=df_plot_male['KL Male'],
    y=df_plot_male['Bottleneck Male'],
    mode='markers',
    name='Males',
    marker=dict(color='red')
))

# Add male regression line trace
fig.add_trace(go.Scatter(
    x=df_plot_male['KL Male'],
    y=np.polyval(male_line, df_plot_male['KL Male']),
    mode='lines',
    name=f'Males (R2 = {male_r2:.2f})',
    line=dict(color='red')
))

# Add female scatter trace
fig.add_trace(go.Scatter(
    x=df_plot_female['KL Female'],
    y=df_plot_female['Bottleneck Female'],
    mode='markers',
    name='Females',
    marker=dict(color='purple')
))

# Add female regression line trace
fig.add_trace(go.Scatter(
    x=df_plot_female['KL Female'],
    y=np.polyval(female_line, df_plot_female['KL Female']),
    mode='lines',
    name=f'Females (R2 = {female_r2:.2f})',
    line=dict(color='purple')
))

# Add labels and title
fig.update_layout(
    title='Comparison between Normalized KL and Normalized Bottleneck',
    xaxis_title='Normalized KL',
    yaxis_title='Normalized Bottleneck',
    plot_bgcolor='rgba(0,0,0,0)',  # this sets the plot background color to transparent
    paper_bgcolor='rgba(0,0,0,0)',  # this sets the paper (around the plot) background color to transparent
)

# Show the plot
fig.show()

fig.write_html("KL_mcmc_vs_bottleneck.html")


#save df_plot_male and df_plot_female as csv
# Save df_plot_male as a CSV file
df_plot_male.to_csv('KLB_MCMC_male.csv', index=False)

# Save df_plot_female as a CSV file
df_plot_female.to_csv('KLB_MCMC_female.csv', index=False)