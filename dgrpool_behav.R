################################################
#
# data_all_pheno is extracted from DGRPpool git hub :
# https://github.com/DeplanckeLab/dgrpool/blob/main/download_phenotypes.R
# the ids of the phenotypes are written as S_XX_XXX
#
#
################################################

# import the rds file from the DGRPpool github
data_all_pheno <-  readRDS("/Users/skumar/Documents/PhD/BrainAnalysis/Behavior/data.all_pheno_21_03_23_filtered.rds") # nolint: line_length_linter.

# print the number of phenotypes
#phenotypes_all <- unique(c(colnames(data_all_pheno[["F"]])[2:ncol(data_all_pheno[["F"]])],  # nolint
#                            colnames(data_all_pheno[["M"]])[2:ncol(data_all_pheno[["M"]])],  # nolint
#                            colnames(data_all_pheno[["NA"]])[2:ncol(data_all_pheno[["NA"]])])) # nolint
#
#message("Number phenotypes = ", length(phenotypes_all)) 
#message("Number phenotypes (by sex) = ", ncol(data_all_pheno[["M"]]) + ncol(data_all_pheno[["F"]]) + ncol(data_all_pheno[["NA"]]) - 3) # nolint

# Extract the data for males, females and NA
data_male <- data_all_pheno[["M"]]
data_female <- data_all_pheno[["F"]]
data_na <- data_all_pheno[["NA"]]

# print number of phenotypes for males
print(colnames(data_male)[2:ncol(data_male)])

# print values of data_male column S1_1315 and print data column dgrp
print(data_male[["S1_1315"]])
print(data_male[["dgrp"]])

# import all DGRP data
#library(data.table)
#data_dgrps <- fread("https://dgrpool.epfl.ch/studies/1/get_file?name=dgrp_lines.tsv&namespace=downloads", sep = "\t", data.table = F) 

#print head of data_dgrps
#head(data_dgrps)


#open phenotypes_to_use.csv
phenotypes_to_use <- read.csv("/Users/skumar/Documents/PhD/BrainAnalysis/Behavior/brain_behavior/phenotypes_to_use.csv")    
# drop the rows having a type_of_behavior different than olfactory, aggresive, and locomotor
phenotypes_to_use <- phenotypes_to_use[phenotypes_to_use$type_of_behavior == " olfactory" | 
phenotypes_to_use$type_of_behavior == " aggresive" | phenotypes_to_use$type_of_behavior == " locomotor"
| phenotypes_to_use$type_of_behavior == " food"
| phenotypes_to_use$type_of_behavior == " sleep"
| phenotypes_to_use$type_of_behavior == " phototaxi",]


# Read all phenotypes from the JSON file
library(jsonlite)
json_phenotypes <- fromJSON("https://dgrpool.epfl.ch/phenotypes.json?all=1")
json_phenotypes <- json_phenotypes[with(json_phenotypes, order(id)),]
rownames(json_phenotypes) <- json_phenotypes$id
message(nrow(json_phenotypes), " phenotypes found")

#print the head of json_phenotypes
head(json_phenotypes)

#iterate through the phenotypes and if the value after "name" is in phenotypes_to_use.csv then add the id in a list called list_id and add name in a name list
list_id <- c()
name <- c()
list_id <- list()
for (i in 1:nrow(json_phenotypes)) {
  if (json_phenotypes[i, "name"] %in% phenotypes_to_use$phenotype) {
    list_id <- c(list_id, json_phenotypes[i, "id"])
    name <- c(name, json_phenotypes[i, "name"])
  }
}

list_id <- unlist(list_id)
print(name)
print(list_id)

# create a new dataframe with the transpose of list_id and name
phenotypes_for_analysis <- data.frame(list_id, name)

# write the dataframe phenotypes_for_analysis to a csv file, without the rows index
write.csv(phenotypes_for_analysis, "/Users/skumar/Documents/PhD/BrainAnalysis/Behavior/brain_behavior/phenotypes_for_analysis.csv", row.names = F)

# change the colnames of data_male to keep only the end 4 characters which are the actual id of the phenotypes
# Get the current column names
current_names <- colnames(data_male)
# Create new column names by keeping only the last four characters
new_names <- substr(current_names, nchar(current_names) - 3, nchar(current_names))
# Assign the new column names to the dataframe
colnames(data_male) <- new_names
print(colnames(data_male))

# change the colnames of data_female to keep only the end 4 characters
# Get the current column names
current_names <- colnames(data_female)
# Create new column names by keeping only the last four characters
new_names <- substr(current_names, nchar(current_names) - 3, nchar(current_names))
# Assign the new column names to the dataframe
colnames(data_female) <- new_names
print(colnames(data_female))

# change the colnames of data_na to keep only the end 4 characters
# Get the current column names
current_names <- colnames(data_na)
# Create new column names by keeping only the last four characters
new_names <- substr(current_names, nchar(current_names) - 3, nchar(current_names))
# Assign the new column names to the dataframe
colnames(data_na) <- new_names
print(colnames(data_na))


#write data_male as csv file
write.csv(data_male, "/Users/skumar/Documents/PhD/BrainAnalysis/Behavior/brain_behavior/data_male.csv", row.names = F)
#write data_female as csv file
write.csv(data_female, "/Users/skumar/Documents/PhD/BrainAnalysis/Behavior/brain_behavior/data_female.csv", row.names = F)
#write data_na as csv file
write.csv(data_na, "/Users/skumar/Documents/PhD/BrainAnalysis/Behavior/brain_behavior/data_na.csv", row.names = F)



