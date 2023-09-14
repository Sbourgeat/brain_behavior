data <-  readRDS("/Users/skumar/Documents/PhD/BrainAnalysis/Behavior/data.all_pheno_21_03_23_filtered.rds") # nolint: line_length_linter.

length(head(data))

column_names <- colnames(data)
print(column_names)
