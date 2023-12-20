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
behaviour = filter(row -> row.exp_type == "operant_place" && row.shock_color == "green" && occursin("dgrp", row.genotype), behaviour)
# Group the DataFrame by genotype and get the distribution of frac_time_on_shockArm for each genotype
distributions = groupby(behaviour, :genotype)

# Create an empty dictionary to store the models
models = Dict()

# Model each genotype
for genotype in keys(distributions)
    # Get the data for this genotype, replace "NA" with missing, remove missing values, and convert to Float64
    data = distributions[genotype].frac_time_on_shockArm
    data = replace(data, "NA" => missing)
    data = collect(skipmissing(data))
    data = parse.(Float64, data)

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
kl_divergence = DataFrame()

# Get the keys of the distributions dictionary as a Vector of strings
keys_str = string.(collect(keys(distributions)))

# Add columns to the DataFrame
for key in keys_str
    kl_divergence[!, key] = Vector{Float64}(undef, length(distributions))
end

# Calculate the KL divergence for each pair of distributions
for (i, genotype_i) in enumerate(keys(models))
    for (j, genotype_j) in enumerate(keys(models))
        # Get the posterior samples of mu and sigma for the two genotypes
        samples_i = DataFrame(models[genotype_i])
        samples_j = DataFrame(models[genotype_j])

        # Estimate the PDFs of the Normal distributions for the two genotypes
        pdf_i = x -> mean(pdf(Normal(mu, sigma), x) for (mu, sigma) in eachrow(samples_i))
        pdf_j = x -> mean(pdf(Normal(mu, sigma), x) for (mu, sigma) in eachrow(samples_j))

        # Calculate the KL divergence between the two PDFs
        kl_divergence[i, j] = sum(pdf_i(x) * (log(pdf_i(x)) - log(pdf_j(x))) for x in -10:0.01:10)
        println("KL divergence between $genotype_i and $genotype_j: $(kl_divergence[i, j])")
    end
end


# Convert the DataFrame to a Matrix
kl_matrix = Matrix(kl_divergence)

# Create a heatmap
heatmap(keys_str, keys_str, kl_matrix, aspect_ratio=1, color=:viridis, clim=(0, maximum(kl_matrix)))