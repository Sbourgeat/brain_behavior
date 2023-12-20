"""
Here we will analyse the distribution of the behaviour 
in each flies and calculate the entropy of the distribution.
Then, we will compare the KL divergence of the distribution 
and correlate it with the bottleneck distances.
"""
import numpy as np  
import pandas as pd 
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.preprocessing import normalize
import pymc as pm
from scipy.interpolate import interp1d
import plotly.graph_objects as go


if __name__ == '__main__':
    # Import summary_all.csv
    behaviour = pd.read_csv("summary_all.csv")

    # Filter the DataFrame to keep only the rows where exp_type == "operant_place", shock_color == 'green', and 'dgrp' is in the genotype column
    behaviour = behaviour[(behaviour['exp_type'] == 'operant_place') & 
                        (behaviour['shock_color'] == 'green') & 
                        (behaviour['genotype'].str.contains('dgrp'))]

    # Group the DataFrame by genotype and get the distribution of frac_time_on_shockArm for each genotype
    distributions = behaviour.groupby('genotype')['frac_time_on_shockArm'].apply(list)

    # Create an empty dictionary to store the models
    models = {}

    # Model each genotype
    #stop_at = 10
    for genotype in distributions.index:
        # Get the data for this genotype
        data = distributions[genotype]

        # Define the model
        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=0, sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=10)

            # Likelihood
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data)

            # Inference
            trace = pm.sample(2000, tune=1000)

        # Store the model and the trace
        models[genotype] = (model, trace)

        #stop_at -= 1
        #if stop_at == 0:
        #    break


    # Create an empty DataFrame to store the KL divergence values
    kl_divergence = pd.DataFrame(index=distributions.index, columns=distributions.index)

    # Calculate the KL divergence for each pair of distributions
    #stop_at = 2


    # Calculate the KL divergence for each pair of distributions
    for i in distributions.index:
        for j in distributions.index:

                # Sample from the posterior distributions to get the inferred distributions
                with models[i][0]:
                    post_pred_i = pm.sample_posterior_predictive(models[i][1])
                with models[j][0]:
                    post_pred_j = pm.sample_posterior_predictive(models[j][1])

                dist_i = post_pred_i.posterior_predictive.likelihood.to_numpy()
                dist_j = post_pred_j.posterior_predictive.likelihood.to_numpy()

                # Normalize the distributions to make them valid probability distributions
                dist_i = dist_i / dist_i.sum()
                dist_j = dist_j / dist_j.sum()

                # Interpolate the distributions to make them the same length
                if len(dist_i) != len(dist_j):
                    x = np.linspace(0, 1, len(dist_i))
                    f = interp1d(x, dist_i, kind='linear', fill_value="extrapolate")
                    x_new = np.linspace(0, 1, len(dist_j))
                    dist_i = f(x_new)

                # Calculate the KL divergence and store it in the DataFrame
                kl_divergence.loc[i, j] = entropy(dist_i, dist_j)
            #stop_at_2 -= 1
            #if stop_at_2 == 0:
            #    break
        
        #stop_at -= 1
        #if stop_at == 0:
        #    break

    # Print the KL divergence matrix
    #print(kl_divergence)



    # Convert the KL divergence DataFrame to a matrix
    # Save kl_divergence as a CSV file
    kl_divergence.to_csv('kl_matrix.csv')
    kl_matrix = kl_divergence.values

    # Create a heatmap
    fig = go.Figure(data=go.Heatmap(
        z=kl_matrix[0][0],
        x=kl_divergence.columns,
        y=kl_divergence.index,
        colorscale='Viridis'))

    # Add labels
    fig.update_layout(
        title='KL Divergence between Genotypes',
        xaxis_title='Genotype',
        yaxis_title='Genotype',
    )

    # Show the plot
    fig.show()