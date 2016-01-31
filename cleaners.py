import numpy as np
import pandas as pd

def example_clean_function():
    def example_clean(dataframe):
        print "cleaning..."
        return dataframe
    return example_clean
    
def integer_sex_mapping():
    def integer_sex_mapping(dataframe):
        dataframe['Gender'] = dataframe['Sex'].map({'female': 0, 'male': 1}).astype(int)
        return dataframe
    return integer_sex_mapping

def clean_missing_embarks_with_mode():
    def clean_missing_embarks_with_mode(dataframe):
        if len(dataframe.Embarked[ dataframe.Embarked.isnull() ]) > 0:
            dataframe.Embarked[ dataframe.Embarked.isnull() ] = \
                        dataframe.Embarked.dropna().mode().values
        return dataframe
    return clean_missing_embarks_with_mode

def integer_embarked_mapping():
    def integer_embarked_mapping(dataframe):
        Ports = list(enumerate(np.unique(dataframe['Embarked'])))
        Ports_dict = { name : i for i, name in Ports }
        dataframe.Embarked = dataframe.Embarked.map( lambda x: Ports_dict[x]).astype(int) 
        return dataframe
    return integer_embarked_mapping

def clean_missing_ages_with_medians():
    def clean_missing_ages_with_medians(dataframe):
        median_age = dataframe['Age'].dropna().median()
        if len(dataframe.Age[ dataframe.Age.isnull() ]) > 0:
            dataframe.loc[ (dataframe.Age.isnull()), 'Age'] = median_age
        return dataframe
    return clean_missing_ages_with_medians

def drop_leftover_features():
    def drop_leftover_features(dataframe):
        dataframe = dataframe.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
        return dataframe
    return drop_leftover_features

def clean_missing_fares_with_medians():
    def clean_missing_fares_with_medians(dataframe):
        if len(dataframe.Fare[ dataframe.Fare.isnull() ]) > 0:
            median_fare = np.zeros(3)
            for f in range(0,3):                                              
                median_fare[f] = dataframe[ dataframe.Pclass == f+1 ]['Fare'].dropna().median()
            for f in range(0,3):            
                dataframe.loc[ (dataframe.Fare.isnull()) & (dataframe.Pclass == f+1 ), 'Fare'] = median_fare[f]
        return dataframe
    return clean_missing_fares_with_medians
