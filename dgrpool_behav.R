################################################
#
# data_all_pheno is extracted from DGRPpool git hub :
# https://github.com/DeplanckeLab/dgrpool/blob/main/download_phenotypes.R
# the ids of the phenotypes are written as S_XX_XXX
#
#
################################################


data_all_pheno <-  readRDS("/Users/skumar/Documents/PhD/BrainAnalysis/Behavior/data.all_pheno_21_03_23_filtered.rds") # nolint: line_length_linter.


# print the number of phenotypes
phenotypes_all <- unique(c(colnames(data_all_pheno[["F"]])[2:ncol(data_all_pheno[["F"]])],  # nolint
                            colnames(data_all_pheno[["M"]])[2:ncol(data_all_pheno[["M"]])],  # nolint
                            colnames(data_all_pheno[["NA"]])[2:ncol(data_all_pheno[["NA"]])])) # nolint

message("Number phenotypes = ", length(phenotypes_all))
message("Number phenotypes (by sex) = ", ncol(data_all_pheno[["M"]]) + ncol(data_all_pheno[["F"]]) + ncol(data_all_pheno[["NA"]]) - 3) # nolint

data_male <- data_all_pheno[["M"]]
data_female <- data_all_pheno[["F"]]
data_na <- data_all_pheno[["NA"]]

# print number of phenotypes for males
print(colnames(data_male)[2:ncol(data_male)])
