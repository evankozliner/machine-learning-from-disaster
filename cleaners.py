import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
    
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

def normalize(X):
    sc = StandardScaler()
    return pd.DataFrame(sc.fit_transform(X))

def get_top_features(dataframe):
    importances_map = []
    X_train = dataframe.iloc[:, range(1, len(dataframe.values[0]))]
    X_train_columns = X_train.columns
    X_train = normalize(X_train)
    X_train.columns = X_train_columns
    feat_labels = X_train.columns
    forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    forest = forest.fit( X_train.values, dataframe.values[0::, 0] )
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        importances_map.append((feat_labels[indices[f]], importances[indices[f]]))
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    return importances_map

def keep_n_top_features(n, features_map, dataframe):
    for i in range(0, len(features_map)):
        if (i >= n):
            dataframe = dataframe.drop([features_map[i][0]], axis=1) 
    return dataframe
        
