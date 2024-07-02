import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the data
dat = pd.read_csv('/Users/skumar/Documents/PhD/BrainAnalysis/Results_Vol_Entropy/macro_meso_140brains.csv')

# Standardize function
def standardize(series):
    return (series - np.mean(series)) / np.std(series)

dat['Volume'] = dat['vol']
# Data preprocessing
dat['Volume'] = standardize(dat['Volume'])
dat['Entropy0'] = standardize(np.exp(dat['Entropy0']))
dat['Entropy1'] = standardize(np.exp(dat['Entropy1']))
dat['Entropy2'] = standardize(np.exp(dat['Entropy2']))
dat['sex'] =dat['sex'].replace({'male': 1, 'female': 2})
dat['DGRP'] = dat['DGRP']

dat = dat.dropna()
dat = dat.sort_values('Volume')

dgrp_means = dat.groupby('DGRP')['Volume'].mean().sort_values()
dat['DGRP_line'] = pd.Categorical(dat['DGRP'], categories=dgrp_means.index, ordered=True)


cc_female = dat[dat['sex'] == 2]
cc_male = dat[dat['sex'] == 1]

# Create the plot
def plot_graph(cc_female, cc_male, dat):
    plt.figure(figsize=(12, 6))
    
    plt.plot(cc_female['DGRP_line'].cat.codes, cc_female['Volume'], 'o', label='Female', color='blue')
    plt.plot(cc_male['DGRP_line'].cat.codes, cc_male['Volume'], 'o', label='Male', color='darkred')
    
    plt.xticks(ticks=np.arange(len(dat['DGRP_line'].cat.categories)), labels=dat['DGRP_line'].cat.categories, rotation=90)
    plt.xlabel('DGRP lines')
    plt.ylabel('Absolute volume (std)')
    plt.legend()
    
    #plt.savefig("volume_plot.jpeg")
    #plt.close()
    plt.show()

plot_graph(cc_female, cc_male, dat)
