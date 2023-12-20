using CSV
using DataFrames
using DataFramesMeta 
using Turing
using StatsBase
using Distributions
using Plots
using StatsFuns


# Import and filter data
behaviour = CSV.read("summary_all.csv", DataFrame)
behaviour = filter(row -> row.exp_type == "operant_place" && row.shock_color == "green" && occursin("dgrp", row.genotype), behaviour)


# Define the model
@model function model(data)
    # Priors
    mu ~ Normal(0, 10)
    sigma ~ truncated(Normal(0, 10), 0, Inf)

    # Likelihood
    for d in data
        d ~ Normal(mu, sigma)
    end
end

# Group by genotype and model each group
genotypes = groupby(behaviour, :genotype)
models = Dict{String, Chains}()

for g in genotypes
    genotype = g.genotype[1]
    data = g.frac_time_on_shockArm
    chain = sample(model(data), NUTS(), 3000)
    models[genotype] = chain
end

# Generate posterior predictive distributions
posterior_predictive = Dict{String, Vector{Float64}}()

for (genotype, chain) in models
    mu_samples = chain[:mu]
    sigma_samples = chain[:sigma]
    predictive_samples = [rand(Normal(mu, sigma)) for (mu, sigma) in zip(mu_samples, sigma_samples)]
    posterior_predictive[genotype] = predictive_samples
end

# Calculate pairwise KL divergence
kl_divergence = Dict{Tuple{String, String}, Float64}()

for (g1, samples1) in posterior_predictive
    for (g2, samples2) in posterior_predictive
        kl = kldivergence(empirical(samples1), empirical(samples2))
        kl_divergence[(g1, g2)] = kl
    end
end

# Convert to Matrix and plot heatmap
genotypes = keys(models)
kl_matrix = [kl_divergence[(g1, g2)] for g1 in genotypes for g2 in genotypes]
kl_matrix = reshape(kl_matrix, length(genotypes), length(genotypes))
#save the kl_matrix as csv
kl_df = DataFrame(kl_matrix, collect(genotypes))
CSV.write("kl_divergence.csv", kl_df)

#heatmap(genotypes, genotypes, kl_df, aspect_ratio=1, color=:viridis, clim=(0, maximum(kl_matrix)))