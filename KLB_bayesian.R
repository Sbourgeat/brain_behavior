# Import the CSV files
female_data <- read.csv("KLB_MCMC_female.csv")
male_data <- read.csv("KLB_MCMC_male.csv")

#scatter plot

# Load the necessary package
library(brms)

# Fit the model for female data
female_model <- brm(KL_Female ~ Bottleneck_Female, data = female_data, family = gaussian())
# Fit the model for male data
male_model <- brm(KL_Male ~ Bottleneck_Male, data = male_data, family = gaussian())


# Load the necessary package
library(tidybayes)

# Extract the posterior samples for the female model
female_samples <- spread_draws(female_model, b_Intercept)

# Extract the posterior samples for the male model
male_samples <- spread_draws(male_model, b_Intercept)

# Combine the samples into a single data frame
combined_samples <- bind_rows(
    mutate(female_samples, group = "Female"),
    mutate(male_samples, group = "Male")
)

# Generate the forest plot
ggplot(combined_samples, aes(x = b_Intercept, y = group)) +
    geom_halfeyeh(fill = "lightblue") +
    theme_minimal()



