# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Import libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def plot(df: pd.DataFrame, columns: list) -> None:
    """
    Create a 3D scatter plot using the specified columns from a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): A list of column names to be used for the scatter plot.

    Returns:
    None

    Raises:
    None
    """

    # Extract the values from the specified columns
    data = df[columns].values

    # Create a scatter plot using plotly
    fig = px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2], color=columns[3])
    fig.update_layout(
        title="Clustering of {}, {}, and {}".format(columns[0], columns[1], columns[2]),
        xaxis_title=columns[0],
        yaxis_title=columns[1],
    )
    fig.show()

# Load the data 
df = pd.read_csv(sys.argv[1])

# Call the function
columns = sys.argv[2:]

plot(df, columns)

