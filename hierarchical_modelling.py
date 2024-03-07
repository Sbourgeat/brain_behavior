"""
Implementation of a hierarchical model in PyMC (4.0).
We will implement 3 levels of the hierarchical model:

    - 1st level: Macro-morpholpogy
    - 2nd level: Meso-morpholpogy
    - 3rd level: Micro-morpholpogy
"""

import numpy
import pymc as pm
import arviz as az
import pandas as pd
import os 
import matplotlib.pyplot as plt

print(f"Running on PyMC v{pm.__version__}")


# Step 1: Load and preprocess the data
data = pd.read_csv("../Results_Vol_Entropy/GWAS_Normalized_EntropyVol.tsv", sep="\t")
#print(data.head(10))

#import behaviour
behav = pd.read_csv("../brain_behavior/formatted_sumamry.csv")
#print(behav.head(10))

# merge data and behav based on DGRP and sex
data = pd.merge(data, behav, on=["DGRP", "sex"])
#print(data.columns)

# Step 2: Define the hierarchical model
with pm.Model() as model:
    # Priors for the parameters
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    coef_morpho = pm.Normal("coef_morpho", mu=0, sigma=10)
    coef_micro = pm.Normal("coef_micro", mu=0, sigma=10)
    coef_meso = pm.Normal("coef_meso", mu=0, sigma=10)

    # Step 3: Define the linear regression model
    mu = intercept + coef_morpho * data["Volume"] + coef_micro * data["Entropy0"] + coef_meso * data["Entropy1"]
    sigma = pm.Exponential("sigma", lam=1)  # Prior for the error term

    # Step 4: Define the likelihood
    likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=data["frac_time_on_shocked"])

    # Step 5: Perform the sampling
    trace = pm.sample(2000, tune=1000, cores=1)  # Use more samples and tune for better results



# mkdir results_pymc
# if the dir does not exist
if not os.path.exists("results_pymc"):
    os.mkdir("results_pymc")

trace.to_netcdf("results_pymc/model_frac_SVM_behav.nc")

# plot the coefficients
axes = az.plot_trace(trace, compact=True)
fig = axes.ravel()[0].figure
fig.savefig('results_pymc/traceplot.png')

# plot posterior
az.plot_energy(trace, show=True)




