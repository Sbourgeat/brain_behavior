import pymc as pm
import pandas as pd
import numpy as np
import arviz
import matplotlib.pyplot as plt
import bambi as bmb


#import data

trace = arviz.from_netcdf("model_frac_SVM_behav.nc")

# Step 6: Examine the results
print(pm.summary(trace))  # Summary statistics of the posterior distributions

arviz.plot_trace(trace)  # Plot the posterior distributions
plt.show()

pm.plot_forest(trace)
plt.show()
