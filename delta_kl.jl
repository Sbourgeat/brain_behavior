using CSV
using DataFrames
using Turing
using StatsBase
using Distributions
using Plots
using Interpolations
using StatsFuns

# Import summary_all.csv
behaviour = CSV.read("summary_all.csv", DataFrame)

# Filter the DataFrame to keep only the rows where exp_type == "operant_place", shock_color == 'green', and 'dgrp' is in the genotype column
behaviour = behaviour[(behaviour.exp_type .== "operant_place") .& 
                      (behaviour.shock_color .== "green") .& 
                      occursin.("dgrp", behaviour.genotype)]

# Group the DataFrame by genotype and get the distribution of frac_time_on_shockArm for each genotype
distributions = groupby(behaviour, :genotype)

# Create an empty dictionary to store the models
models = Dict()

# Model each genotype
for genotype in keys(distributions)
    # Get the data for this genotype
    data = distributions[genotype].frac_time_on_shockArm

    # Define the model
    @model function model(data)
        # Priors
        mu ~ Normal(0, 10)
        sigma ~ truncated(Normal(0, 10), 0, Inf)

        # Likelihood
        data ~ Normal(mu, sigma)
    end

    # Inference
    chain = sample(model(data), NUTS(), 3000)

    # Store the model and the chain
    models[genotype] = chain
end

# Create an empty DataFrame to store the KL divergence values
kl_divergence = DataFrame(Matrix{Float64}(undef, length(distributions), length(distributions)), keys(distributions), keys(distributions))

# Calculate the KL divergence for each pair of distributions
for (i, genotype_i) in enumerate(keys(distributions))
    for (j, genotype_j) in enumerate(keys(distributions))
        # Sample from the posterior distributions to get the inferred distributions
        post_pred_i = rand(models[genotype_i], 1000)
        post_pred_j = rand(models[genotype_j], 1000)

        # Normalize the distributions to make them valid probability distributions
        dist_i = normalize(post_pred_i, 1)
        dist_j = normalize(post_pred_j, 1)

        # Interpolate the distributions to make them the same length
        if length(dist_i) != length(dist_j)
            x = range(0, stop=1, length=length(dist_i))
            f = LinearInterpolation(x, dist_i, extrapolation_bc=Line())
            x_new = range(0, stop=1, length=length(dist_j))
            dist_i = f(x_new)
        end

        # Calculate the KL divergence and store it in the DataFrame
        kl_divergence[i, j] = kl_div(dist_i, dist_j)
    end
end

# Save kl_divergence as a CSV file
CSV.write("kl_matrix.csv", kl_divergence)

# Convert the KL divergence DataFrame to a matrix
kl_matrix = Matrix(kl_divergence)

# Create a heatmap
heatmap(kl_divergence.columns, kl_divergence.index, kl_matrix, title="KL Divergence between Genotypes", xlabel="Genotype", ylabel="Genotype")