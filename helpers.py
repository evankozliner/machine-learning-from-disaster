import pandas as pd
import cleaners as c
from subprocess import call
import numpy as np
import csv as csv
from subprocess import call
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def cross_validate(training_data, classifier):
    target = extract_target(training_data)
    #scores = cross_val_score(classifier, training_data, target, cv=5, n_jobs=1)
    #print scores
    #print scores.mean()
    X_train, X_test, y_train, y_test = train_test_split(
            training_data, target , test_size=.90 , random_state=0)
    classifier.fit(X_train, y_train)
    scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    #y_pred = classifier.predict(X_test)
    #no_samples = len(y_test)
    #print('Misclassified samples: %d of %d' % ((y_test != y_pred).sum() , no_samples))
    #print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Extracts a target for cross validation
def extract_target(data):
    #results = pd.DataFrame(data).iloc[:, 0].values
    #return results
    return data.iloc[:, 0].values

def clean(data_csv, cleaning_functions):
    dataframe = pd.read_csv(data_csv, header=0)
    ids = dataframe['PassengerId'] # Return ids seperately before they're dropped
    for cleaner in cleaning_functions:
        dataframe = cleaner()(dataframe)
    return ids.values, dataframe

def write_predictions(ids, predictions, filename):
    predictions_file = open(filename, "wb")
    file_obj = csv.writer(predictions_file)
    file_obj.writerow(["PassengerId", "Survived"])
    file_obj.writerows(zip(ids, predictions))
    predictions_file.close()
    archive()

def archive():
    call(['./generate_archive.sh'])
