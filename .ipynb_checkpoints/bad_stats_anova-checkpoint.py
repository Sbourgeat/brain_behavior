import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contrast import ContrastResults

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
    #dat['sex'] = dat['sex'].replace({'male': 1, 'female': 2})
    dat['DGRP'] = dat['DGRP']

    dat.replace([np.inf, -np.inf], np.nan, inplace=True)
    dat = dat.dropna()
    dat = dat.sort_values('Volume')

    #print(dat['DGRP'],type(dat['DGRP']))

    dat['DGRP'] = dat['DGRP'].astype('category')
    dat['sex'] = dat['sex'].astype('category')

    # Define the formula for the ANOVA model with interaction
    formula = 'vol ~ C(DGRP) * C(sex)'

    # Fit the model
    model = smf.ols(formula, data=dat).fit()

    # Perform ANOVA
    anova_results = anova_lm(model, typ=2)

    # Print the results
    print("ANOVA results with interaction:")
    print(anova_results)

if __name__ == '__main__':
    main()
