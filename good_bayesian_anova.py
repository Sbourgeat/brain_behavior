import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az

def main():
        
# Load the data
    dat = pd.read_csv('/Users/skumar/Documents/PhD/BrainAnalysis/Results_Vol_Entropy/macro_meso_140brains.csv')

# Standardize function
    def standardize(series):
        return (series - np.mean(series)) / np.std(series)

# Data preprocessing
    dat['Volume'] = standardize(dat['vol'])
    dat['Entropy0'] = standardize(np.exp(dat['Entropy0']))
    dat['Entropy1'] = standardize(np.exp(dat['Entropy1']))
    dat['Entropy2'] = standardize(np.exp(dat['Entropy2']))
    dat['sex'] = dat['sex'].replace({'male': 1, 'female': 2})
    dat['DGRP'] = dat['DGRP']
# Replace inf values with NaN and drop them
    dat.replace([np.inf, -np.inf], np.nan, inplace=True)
    dat = dat.dropna()

# Sort values
    dat = dat.sort_values('Volume')

# Create the DGRP_line column as a Categorical type and reorder categories
    dgrp_means = dat.groupby('DGRP')['Volume'].mean().sort_values()
    dat['DGRP_line'] = pd.Categorical(dat['DGRP'], categories=dgrp_means.index, ordered=True)

# Define the formula for the Bayesian model
    formula = 'Volume ~ C(DGRP_line) + C(sex) + Entropy0 + Entropy1 + Entropy2'
    #formula_interaction = 'Volume ~ C(DGRP_line) * C(sex) + Entropy0 * Entropy1 * Entropy2'

# Fit the Bayesian models
    model = bmb.Model(formula, data=dat)
    #model_interaction = bmb.Model(formula_interaction, data=dat)

# Inference
    trace = model.fit(draws=2000, tune=1000, cores=2)
    #trace_interaction = model_interaction.fit(draws=2000, tune=1000, cores=2)

# Summary of the results
    print("\nSummary of the Bayesian model without interaction:")
    az.summary(trace, hdi_prob=0.95)

    #print("\nSummary of the Bayesian model with interaction:")
    #az.summary(trace_interaction, hdi_prob=0.95)

# Posterior Predictive Checks
    ppc = model.predict(trace)
    #ppc_interaction = model_interaction.predict(trace_interaction)

# Plot Posterior Predictive Checks
    az.plot_ppc(ppc)
    plt.show()
    az.plot_ppc(ppc_interaction)
    plt.show()
# Define and print contrasts (equivalent to comparing parameters)
    contrast_results_entropy0 = az.plot_forest(trace, var_names=["Entropy0"], combined=True)
    contrast_results_entropy1 = az.plot_forest(trace, var_names=["Entropy1"], combined=True)
    contrast_results_entropy2 = az.plot_forest(trace, var_names=["Entropy2"], combined=True)

if __name__ == "__main__":
    main()
