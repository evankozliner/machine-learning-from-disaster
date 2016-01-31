import pandas as pd
import numpy as np
import csv as csv
from subprocess import call

def build_and_clean(data_csv, cleaning_functions):
    dataframe = pd.read_csv(data_csv, header=0)
    ids = dataframe['PassengerId'] # Return ids seperately before they're dropped
    for cleaner in cleaning_functions:
        dataframe = cleaner()(dataframe)
    print dataframe
    return ids.values, dataframe.values

def write_predictions(ids, predictions, filename):
    predictions_file = open(filename, "wb")
    file_obj = csv.writer(predictions_file)
    file_obj.writerow(["PassengerId", "Survived"])
    file_obj.writerows(zip(ids, predictions))
    predictions_file.close()

