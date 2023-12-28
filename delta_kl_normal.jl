using CSV
using DataFrames
using DataFramesMeta 
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
behaviour.ls = parse.(Float64, behaviour.ls)

# Group by genotype and estimate the KDE for each group
genotypes = groupby(behaviour, :genotype)
kdes = Dict{String, UnivariateKDE}()

for g in genotypes
    genotype = g.genotype[1]
    data = g.ls
    kdes[genotype] = kde(data)
end

# Calculate pairwise KL divergence
kl_divergence = Dict{Tuple{String, String}, Float64}()

epsilon = 1e-10

for (g1, kde1) in kdes
    for (g2, kde2) in kdes
        kl = quadgk(x -> (pdf(kde1, x) + epsilon) * log((pdf(kde1, x) + epsilon) / (pdf(kde2, x) + epsilon)), -Inf, Inf)[1]
        kl_divergence[(g1, g2)] = kl
    end
end

# Convert to Matrix and plot heatmap
genotypes = keys(kdes)
kl_matrix = [kl_divergence[(g1, g2)] for g1 in genotypes for g2 in genotypes]
kl_matrix = reshape(kl_matrix, length(genotypes), length(genotypes))

# Save the kl_matrix as csv
kl_df = DataFrame(kl_matrix, collect(genotypes))
CSV.write("kl_divergence_kde.csv", kl_df)