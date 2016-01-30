import pandas as pd
import numpy as np
import csv as csv

def build_and_clean(data_csv, cleaning_functions):
    dataframe = pd.read_csv(data_csv, header=0)
    for cleaner in cleaning_functions:
        dataframe = cleaner(dataframe)
    return dataframe 
