# Load necessary libraries
library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(brms)
# Load necessary library
library(posterior)

library(entropy)
library(heatmaply)
library(webshot)
library(bayesplot)



# Import and filter data
behaviour <- read_csv("summary_with_learningscore_pl_green_70genotypes.csv")
behaviour <- behaviour %>% filter(exp_type == "operant_place" & shock_color == "green" & grepl("dgrp", genotype))

# Remove rows where ls is NA
behaviour$ls <- ifelse(behaviour$ls == "NA", NA, behaviour$ls)
behaviour <- behaviour %>% drop_na(ls)

# Convert the column to numeric
behaviour$ls <- as.numeric(behaviour$ls)

# Calculate the mean and standard deviation of the column
mean_ls <- mean(behaviour$ls, na.rm = TRUE)
std_ls <- sd(behaviour$ls, na.rm = TRUE)

# Z-score normalize the column
behaviour$ls <- (behaviour$ls - mean_ls) / std_ls

# Group by genotype and model each group
genotypes <- behaviour %>% group_by(genotype)

# Define the model
model <- function(data) {
    fit <- brm(
        bf(data ~ 1, family = gaussian()),  # Normal prior
        data = data.frame(data = data),  # Create a data frame on the fly
        prior = prior(normal(0, 10), class = "Intercept"),  # Normal prior on the intercept
        iter = 2000,  # Number of MCMC iterations
        chains = 4  # Number of MCMC chains
    )
    return(fit)
}

# Model each group#####################################################
models <- genotypes %>% summarise(model = list(model(ls)))################### long to run!!!
###########################################################################



# Perform a posterior predictive check for each model
models$model %>% purrr::map(~brms::pp_check(.))

# Calculate the LOO for each model
models$model %>% purrr::map(~brms::loo(.))

# Extract the posterior predictive distribution for each genotype
posterior_predictive <- lapply(models$model, posterior_predict, nsamples = 1000)



# Define a function to discretize a continuous distribution
discretize <- function(x, n_bins = 100) {
    cut(x, breaks = n_bins, labels = FALSE) / n_bins
}

# Discretize and normalize the posterior predictive distributions
posterior_predictive_discrete <- lapply(posterior_predictive, discretize)

# Define a function to calculate the KL divergence between two discrete distributions
kl_divergence <- function(p, q) {
    KL.empirical(p, q)
}

# Calculate pairwise KL divergence
kl_matrix <- matrix(0, nrow = length(posterior_predictive_discrete), ncol = length(posterior_predictive_discrete))

for (i in 1:length(posterior_predictive_discrete)) {
    for (j in 1:length(posterior_predictive_discrete)) {
        kl_matrix[i, j] <- kl_divergence(posterior_predictive_discrete[[i]], posterior_predictive_discrete[[j]])
    }
}

# save the kl matrix
write.csv(kl_matrix, "kl_matrix_new_working.csv", row.names = TRUE)

# Get the unique genotypes
genotype_names <- unique(behaviour$genotype)

# Set the column and row names of the KL divergence matrix
colnames(kl_matrix) <- genotype_names
rownames(kl_matrix) <- genotype_names

# Print the KL divergence matrix
print(kl_matrix)

# Start the graphics device
png("results/kl_divergence.png", width = 500, height = 500, res = 100)

# Plot the heatmap
heatmap(kl_matrix, xlab = "Genotype", ylab = "Genotype", main = "Pairwise KL Divergence")

# Stop the device and save the plot
dev.off()

# Convert the matrix to a data frame
kl_df <- as.data.frame(as.table(kl_matrix))

# Create the heatmap
p <- ggplot(kl_df, aes(x = Var1, y = Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Genotype", y = "Genotype", fill = "KL Divergence")

plot(p)


# Create the heatmap with a dendrogram
heatmaply(kl_matrix, 
          xlab = "Genotype", 
          ylab = "Genotype", 
          main = "Pairwise KL Divergence", 
          colors = colorRampPalette(c("white", "red"))(255),
          dendrogram = "both")



# Install PhantomJS (required by webshot)
#webshot::install_phantomjs()

# Create the heatmap with a dendrogram
p <- heatmaply(kl_matrix, 
               xlab = "Genotype", 
               ylab = "Genotype", 
               main = "Pairwise KL Divergence", 
               colors = colorRampPalette(c("white", "red"))(255),
               dendrogram = "both",
               file = "heatmap.html")

# Save the heatmap as a PNG image
webshot("heatmap.html", file = "results/heatmap.html")
