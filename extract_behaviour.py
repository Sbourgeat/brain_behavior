#!/usr/bin/env python3

import pandas as pd

# import data from data_male.csv and data_female.csv
def import_data():
    data_male = pd.read_csv("data_male.csv")
    data_female = pd.read_csv("data_female.csv")
    return data_male, data_female


# import volume and entropy data from vol_entropy.csv
def import_vol_entropy():
    vol_entropy = pd.read_csv("vol_entropy.csv")
    return vol_entropy


