using CSV
using DataFrames
using DataFramesMeta 
using Turing
using StatsBase
using Distributions
using Plots
using StatsFuns
using KernelDensity
using QuadGK
using Statistics

# Import and filter data
behaviour = CSV.read("summary_with_learningscore_pl_green_70genotypes.csv", DataFrame)
behaviour = filter(row -> row.exp_type == "operant_place" && row.shock_color == "green" && occursin("dgrp", row.genotype), behaviour)

# Remove rows where ls is NA
behaviour.ls = replace(behaviour.ls, "NA" => missing)

behaviour = dropmissing(behaviour, :ls)
# Convert the column to Float64
behaviour.ls = parse.(Float64, behaviour.ls)
# Calculate the mean and standard deviation of the column
mean_ls = mean(behaviour.ls) 
std_ls = std(behaviour.ls)

# Z-score normalize the column
behaviour.ls = (behaviour.ls .- mean_ls) ./ std_ls

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
# Filter the DataFrame for 'dgrp584'
dgrp584_df = genotypes[1]

# Plot the 'ls' values


for g in genotypes
    genotype = g.genotype[1]
    data = g.ls
    chain = sample(model(data), NUTS(), 3000)
    models[genotype] = chain
end

# Generate posterior predictive distributions
posterior_predictive = Dict{String, Vector{Float64}}()


posterior_predictive_pdf = Dict{String, Function}()


for (genotype, chain) in models
    mu_samples = chain[:mu]
    sigma_samples = chain[:sigma]
    predictive_samples = [rand(Normal(mu, sigma)) for (mu, sigma) in zip(mu_samples, sigma_samples)]
    predictive_samples = reshape(predictive_samples, :)
    posterior_predictive[genotype] = predictive_samples

    # Estimate a KDE from the samples
    kde_est = kde(predictive_samples)

    # Store the PDF of the KDE
    posterior_predictive_pdf[genotype] = x -> pdf(kde_est, x)
end


# Get the PDF for "dgrp584"
pdf_dgrp584 = posterior_predictive_pdf["dgrp584"]

# Generate a range of x values
x_values = range(minimum(posterior_predictive["dgrp584"]), maximum(posterior_predictive["dgrp584"]), length=1000)

# Calculate the PDF values for the range of x values
pdf_values = pdf_dgrp584.(x_values)

# Plot the PDF
plot(x_values, pdf_values, title="PDF of Posterior Predictive Distribution for dgrp584", label="")



# Calculate pairwise KL divergence
kl_divergence = Dict{Tuple{String, String}, Float64}()

epsilon = 1e-10  # small constant to avoid taking log of zero

for (g1, kde1) in posterior_predictive_pdf
    for (g2, kde2) in posterior_predictive_pdf
        # Define the integrand for the KL divergence with log smoothing
        integrand = x -> kde1(x) * (log(kde1(x) + epsilon) - log(kde2(x) + epsilon))

        # Calculate the range of x values where the PDFs of kde1 and kde2 are non-zero
        x_min = min(minimum(posterior_predictive[g1]), minimum(posterior_predictive[g2]))
        x_max = max(maximum(posterior_predictive[g1]), maximum(posterior_predictive[g2]))

        # Calculate the KL divergence using numerical integration
        kl, err = quadgk(integrand, x_min, x_max)

        kl_divergence[(g1, g2)] = kl
    end
end

# Convert to Matrix and plot heatmap

genotypes = keys(models)
kl_matrix = [kl_divergence[(g1, g2)] for g1 in genotypes for g2 in genotypes]
kl_matrix = reshape(kl_matrix, length(genotypes), length(genotypes))
#save the kl_matrix as csv
kl_df = DataFrame(kl_matrix, collect(genotypes))
CSV.write("kl_divergence_pp.csv", kl_df)

#heatmap(genotypes, genotypes, kl_df, aspect_ratio=1, color=:viridis, clim=(0, maximum(kl_matrix)))









# Calculate the KL divergence between "dgrp584" and "dgrp721"
kl_dgrp584_dgrp721 = kl_divergence[("dgrp584", "dgrp721")]