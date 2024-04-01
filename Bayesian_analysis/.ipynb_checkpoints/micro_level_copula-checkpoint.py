import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr
from scipy.stats import zscore
from scipy.stats import expon, multivariate_normal, norm
from sklearn.cross_decomposition import CCA


# Environment settings for reproducibility
#az.style.use(["arviz-white", "arviz-purplish"])
az.style.use("arviz-darkgrid")
warnings.filterwarnings("ignore", module="scipy")
print(f"Running on PyMC v{pm.__version__}") # Running on PyMC v5.10.4
RANDOM_SEED = 8924


def canonical_correlation(a,b):
    cca = CCA(n_components=1)
    cca.fit(a, b)
    canonical_a, canonical_b = cca.transform(a, b)
    return canonical_a, canonical_b

def standardization(data):
    return (zscore(data))

def joint_plot(data, var1, var2, var_names):
    sns.axes_style("white")
    sns.set(font_scale=2)
    sns.jointplot(x=var1, y=var2, data=data, kind="kde")
    plt.savefig(f"Bayesian_analysis/Results/joint_distribution_{var_names[0]}_{var_names[1]}.png")


def transform_data(marginal_idata, var1, var2):
    # point estimates
    a_mu = marginal_idata.posterior["a_mu"].mean().item()
    a_sigma = marginal_idata.posterior["a_sigma"].mean().item()
    b_mu = marginal_idata.posterior["b_mu"].mean().item()
    b_sigma = marginal_idata.posterior["b_sigma"].mean().item()
    # transformations from observation space -> uniform space
    __a = pt.exp(pm.logcdf(pm.Normal.dist(mu=a_mu, sigma=a_sigma), var2))
    __b = pt.exp(pm.logcdf(pm.Normal.dist(mu=b_mu, sigma=b_sigma), var1))
    # uniform space -> multivariate normal space
    _a = pm.math.probit(__a)
    _b = pm.math.probit(__b)
    # join into an Nx2 matrix
    data = pt.math.stack([_a, _b], axis=1).eval()
    return data, a_mu, a_sigma, b_mu, b_sigma

def copula (data, var1, var2, var_names):
    coords = {"obs_id": np.arange(len(var1))}
    print("Modelling the marginals")
    with pm.Model(coords=coords) as marginal_model:
        """
        Assumes observed data in variables `a` and `b`
        """
        # hypreprior
        #lambda_b = pm.HalfNormal('lambda_b', sigma = 10)
        # marginal estimation
        a_mu = pm.Normal("a_mu", mu=0, sigma=10)
        a_sigma = pm.Exponential("a_sigma", lam=0.5)
        pm.Normal("a", mu=a_mu, sigma=a_sigma, observed=var2, dims="obs_id")

        b_mu = pm.Normal("b_mu", mu=0, sigma=10)
        b_sigma = pm.Exponential("b_sigma", lam=0.5 )
        pm.Normal("b", mu = b_mu , sigma = b_sigma , observed=var1, dims="obs_id")


    #marginal posterior
    with marginal_model:
        marginal_idata = pm.sample(5000, tune=2000, chains=10, cores=-1 ,random_seed=RANDOM_SEED)

    axs = az.plot_trace(marginal_idata)
    fig = axs.ravel()[0].figure
    fig.savefig(f"Bayesian_analysis/Results/marignal_traces_{var_names[0]}_{var_names[1]}.png")
    
    
    axs = az.plot_posterior(
    marginal_idata, var_names=["a_mu", "a_sigma", "b_mu", "b_sigma"]);
    fig = axs.ravel()[0].figure
    fig.savefig(f"Bayesian_analysis/Results/marignal_posterior_{var_names[0]}_{var_names[1]}.png")


    #transform data 
    data, a_mu, a_sigma, b_mu, b_sigma = transform_data(marginal_idata, var1, var2)

    # copula model 
    print("Modelling the copula")
    coords.update({"param": ["a", "b"], "param_bis": ["a", "b"]})
    with pm.Model(coords=coords) as copula_model:
        # Prior on covariance of the multivariate normal
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol",
            n=2,
            eta=2.0,
            sd_dist=pm.Exponential.dist(1.0),
            compute_corr=True,
        )
        cov = pm.Deterministic("cov", chol.dot(chol.T), dims=("param", "param_bis"))

        # Likelihood function
        pm.MvNormal("N", mu=0.0, cov=cov, observed=data, dims=("obs_id", "param"))

    with copula_model:
        copula_idata = pm.sample(5000, chains=10,random_seed=RANDOM_SEED, tune=2000, cores=-1)
    
    axs = az.plot_trace(copula_idata, var_names=["cov"])
    fig = axs.ravel()[0].figure
    fig.savefig(f"Bayesian_analysis/Results/copula_traces_{var_names[0]}_{var_names[1]}.png")
    
    axs = az.plot_posterior(copula_idata, var_names=["cov"])
    fig = axs.ravel()[0].figure
    fig.savefig(f"Bayesian_analysis/Results/copula_posterior_{var_names[0]}_{var_names[1]}.png")

    with copula_model:
        pm.compute_log_likelihood(copula_idata)
    

    # data munging to extract covariance estimates from copula_idata in useful shape
    d = {k: v.values.reshape((-1, *v.shape[2:])) for k, v in copula_idata.posterior[["cov"]].items()}

    # generate (a, b) samples
    ab = np.vstack([multivariate_normal([0, 0], cov).rvs() for cov in d["cov"]])

    # transform to uniform space
    uniform_a = norm().cdf(ab[:, 0])
    uniform_b = norm().cdf(ab[:, 1])

    # transform to observation space
    # estimated marginal parameters a_mu, a_sigma, b_mu, b_sigma are point estimates from marginal estimation.
    ppc_a = norm(loc=a_mu, scale=a_sigma).ppf(uniform_a)
    ppc_b = norm(loc=b_mu, scale=b_sigma).ppf(uniform_b)
    
    # plot inferences in red
    #az.style.use(["arviz-white", "arviz-purplish"])
    axs = az.plot_pair(
        {f"{var_names[0]}": ppc_b, f"{var_names[1]}": ppc_a},
        marginals=True,
        # kind=["kde", "scatter"],
        kind="kde",
        scatter_kwargs={"alpha": 0.01},
        kde_kwargs=dict(contour_kwargs=dict(colors="r", linestyles="-"), contourf_kwargs=dict(alpha=0)),
        marginal_kwargs=dict(color="r"),
        textsize=25
    );
    fig = axs.ravel()[0].figure
    fig.savefig(f'Bayesian_analysis/Results/joint_estimate_distribution_{var_names[0]}_{var_names[1]}.png')


    # regression model 
    print("Modelling the regression")

    with pm.Model() as reg_model:
        xdata = pm.ConstantData("x", ppc_b, dims="obs_id")

        # define priors
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        slope = pm.Normal("slope", mu=0, sigma=1)
        sigma = pm.HalfCauchy("sigma", beta=10)

        mu = pm.Deterministic("mu", intercept + slope * xdata, dims="obs_id")

        # define likelihood
        likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=ppc_a, dims="obs_id")

        # inference
        trace = pm.sample(1000, chains=10,random_seed=RANDOM_SEED, tune=2000, cores=-1)

    with reg_model:
        pm.compute_log_likelihood(trace)
    
    axs = az.plot_posterior(trace, var_names=['intercept', 'slope', 'sigma'])
    fig = axs.ravel()[0].figure
    fig.savefig(f"Bayesian_analysis/Results/regression_posterior_{var_names[0]}_{var_names[1]}.png")



    plt.figure()
    post = az.extract(trace, num_samples=20)
    x_plot = xr.DataArray(np.linspace(ppc_b.min(), ppc_b.max(), 100), dims="plot_id")
    lines = post["intercept"] + post["slope"] * x_plot

    plt.scatter(ppc_b, ppc_a)
    plt.plot(x_plot, lines.transpose(), alpha=0.4, color="C1")
    plt.xlabel(f"{var_names[0]}")
    plt.ylabel(f"{var_names[1]}")
    #plt.title("Posterior predictive for normal likelihood");
    plt.savefig(f'Bayesian_analysis/Results/regression_prediction_{var_names[0]}_{var_names[1]}.png')


if __name__ == "__main__":
    # load and preprocess the data 
    # Step 1: Load and preprocess the data
    #brain_morpho = pd.read_csv("classical_descriptor.csv", sep=",")
    brain_morpho = pd.read_csv("entropy_vol_sep2023.csv", sep=",")
    
    #brain_morpho = pd.read_csv("Bayesian_analysis/canonical_data.csv", sep=',')

    #print(data.head(10))

    #import behaviour
    behav = pd.read_csv("dgrpool_kl_behaviour.tsv", sep='\t')

    # drop missing values
    behav = behav.dropna()
    # transform brain_morpho['DGRP'] into string
    brain_morpho['DGRP'] = brain_morpho['DGRP'].astype(str)

    # change column name Sex by sex
    brain_morpho = brain_morpho.rename(columns={'Sex': 'sex'})
    # format correctly brain morphology
    #brain_morpho[['DGRP', 'sex']] = brain_morpho['DGRP'].str.split('_', expand=True).get([0, 1])

    # Add 'DGRP_' in front of each 'XXX'
    brain_morpho['DGRP'] = 'DGRP_' + brain_morpho['DGRP'].str.zfill(3)

    # Replace 'female' and 'male' with 'F' and 'M' in the 'sex' column
    brain_morpho['sex'] = brain_morpho['sex'].map({'female': 'F', 'male': 'M'})
   
    print(brain_morpho.head())
    input('Press Enter to continue...')

    # merge data and behav based on DGRP and sex
    full_data = pd.merge(brain_morpho, behav, on = ['DGRP', 'sex'])
    
    print(full_data.head())
    input('Press Enter to continue...')
    

    # standardize the data

    var1 = standardization(full_data[['entropy0', 'entropy1', 'entropy2']])
    var2 = standardization(full_data['KL_frac'])

    # extract the canonical vector of entropies
    var1, _ =  canonical_correlation(var1, var2)
    var1 = var1[:,0]
    #print(var1)
    full_data['mesomorphology'] = var1
    print(full_data.head())
    input('Press Enter to continue...')


    var_names = ['mesomorphology', 'KL_frac']
    
    # observation joint plot
    joint_plot(full_data, var1, var2, var_names)

    # run the copula
    copula(full_data, var1, var2, var_names)



