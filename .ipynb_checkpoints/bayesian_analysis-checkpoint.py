import pymc as pm
import pandas as pd
import numpy as np
import arviz
import matplotlib.pyplot as plt
import bambi as bmb

"""

# Step 1: Load and preprocess the data
data = pd.read_csv('data.csv') # Replace 'your_data.csv' with your actual data file
# Assuming the columns are 'frac_time_on_shocked', 'Volume', 'activity', and 'correct_choices'
frac_time = data['frac_time_on_shocked']
volume = data['Volume']
activity = data['activity']
correct_choices = data['correct_choices']

# Step 2: Define the prior distributions
with pm.Model() as model:
    # Priors for the parameters
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    coef_volume = pm.Normal('coef_volume', mu=0, sigma=10)
    coef_activity = pm.Normal('coef_activity', mu=0, sigma=10)
    coef_correct_choices = pm.Normal('coef_correct_choices', mu=0, sigma=10)
    
    # Step 3: Define the linear regression model
    mu = intercept + coef_volume * volume + coef_activity * activity + coef_correct_choices * correct_choices
    sigma = pm.Exponential('sigma', lam=1)  # Prior for the error term
    
    # Step 4: Define the likelihood
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=frac_time)
    
    # Step 5: Perform the sampling
    trace = pm.sample(2000, tune=1000, cores=1)  # Use more samples and tune for better results


trace.to_netcdf("model_frac.nc")

""" 

"""
data = pd.read_csv('data.csv') # Replace 'your_data.csv' with your actual data file
# Assuming the columns are 'frac_time_on_shocked', 'Volume', 'activity', and 'correct_choices'
frac_time = data['frac_time_on_shocked']
volume = data['Volume']
activity = data['activity']
correct_choices = data['correct_choices']
scores = data['scores']

# Step 2: Define the prior distributions
with pm.Model() as model:
    # Priors for the parameters
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    coef_scores = pm.Normal('coef_scores', mu=0, sigma=10)
    coef_activity = pm.Normal('coef_activity', mu=0, sigma=10)
    coef_correct_choices = pm.Normal('coef_correct_choices', mu=0, sigma=10)
    
    # Step 3: Define the linear regression model
    mu = intercept + coef_scores * scores + coef_activity * activity + coef_correct_choices * correct_choices
    sigma = pm.Exponential('sigma', lam=1)  # Prior for the error term
    
    # Step 4: Define the likelihood
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=frac_time)
    
    # Step 5: Perform the sampling
    trace = pm.sample(2000, tune=1000, cores=1)  # Use more samples and tune for better results


trace.to_netcdf("model_frac_SVM.nc")



"""


"""
data = pd.read_csv('data.csv') # Replace 'your_data.csv' with your actual data file
# Assuming the columns are 'frac_time_on_shocked', 'Volume', 'activity', and 'correct_choices'
frac_time = data['frac_time_on_shocked']
volume = data['Volume']
entropy0 = data["Entropy0"]
entropy1 = data["Entropy1"]
entropy2 = data["Entropy2"]
activity = data['activity']
correct_choices = data['correct_choices']
scores = data['scores']

# Step 2: Define the prior distributions
with pm.Model() as model:
    # Priors for the parameters
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    coef_scores = pm.Normal('coef_scores', mu=0, sigma=10)
    coef_activity = pm.Normal('coef_activity', mu=0, sigma=10)
    coef_correct_choices = pm.Normal('coef_correct_choices', mu=0, sigma=10)
    coeff_volume = pm.Normal('coef_volume', mu=0, sigma=10)
    coeff_entropy0 = pm.Normal('coeff_entropy0', mu=0, sigma=10)
    coeff_entropy1 = pm.Normal('coeff_entropy1', mu=0, sigma=10)
    coeff_entropy2 = pm.Normal('coeff_entropy2', mu=0, sigma=10)
    # Step 3: Define the linear regression model
    mu = intercept + coef_scores * scores + coef_activity * activity + coef_correct_choices * correct_choices
    sigma = pm.Exponential('sigma', lam=1)  # Prior for the error term
    
    # Step 4: Define the likelihood
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=frac_time)
    
    # Step 5: Perform the sampling
    trace = pm.sample(2000, tune=1000, cores=1)  # Use more samples and tune for better results


trace.to_netcdf("model_frac_SVM.nc")

"""


data = pd.read_csv('data_SVM.csv') # Replace 'your_data.csv' with your actual data file
# Assuming the columns are 'frac_time_on_shocked', 'Volume', 'activity', and 'correct_choices'
morpho = data["scores"]
behav = data["scores2"]
# Step 2: Define the prior distributions
with pm.Model() as model:
    # Priors for the parameters
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    coef_morpho = pm.Normal('coef_scores', mu=0, sigma=10)
    #coef_activity = pm.Normal('coef_activity', mu=0, sigma=10)
     # Step 3: Define the linear regression model
    mu = intercept + coef_morpho * morpho
    sigma = pm.Exponential('sigma', lam=1)  # Prior for the error term
    
    # Step 4: Define the likelihood
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=behav)
    
    # Step 5: Perform the sampling
    trace = pm.sample(2000, tune=1000, cores=1)  # Use more samples and tune for better results


trace.to_netcdf("model_frac_SVM_behav.nc")


