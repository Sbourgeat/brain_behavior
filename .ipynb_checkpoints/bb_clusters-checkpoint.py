# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Import libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
import plotly.express as px

def cluster_and_plot(df, columns):
    """
    Clusters the values of two columns of a dataframe using affinity propagation and plots the clusters.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    column1 (str): The name of the first column.
    column2 (str): The name of the second column.
    """

    # Importimport plotly.express as px

# Extract the values from the specified columns
    data = df[columns].values

# Perform affinity propagation clustering
    clustering = AffinityPropagation().fit(data)
    labels = clustering.labels_

# Create a scatter plot using plotly
    fig = px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2], color=labels)
    fig.update_layout(
        title="Clustering of {}, {}, and {}".format(columns[0], 
                                                    columns[1], columns[2]),
        xaxis_title=columns[0],
        yaxis_title=columns[1],
    )
    fig.show()

# Load the data 
df = pd.read_csv(sys.argv[1])

# Call the function
columns = sys.argv[2:]

cluster_and_plot(df, columns)
