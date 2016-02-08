import helpers
import cleaners as c
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

# TODO can't clean with the exact same function because I'm dropping the worst features
cleaning_functions = [c.integer_sex_mapping, 
        c.clean_missing_embarks_with_mode,
        c.integer_embarked_mapping,
        c.clean_missing_ages_with_medians,
        c.clean_missing_fares_with_medians,
        c.drop_leftover_features,
        c.use_only_top_features]

train_ids, training_data = helpers.clean('data/train.csv', cleaning_functions)
test_ids, test_data = helpers.clean('data/test.csv', cleaning_functions)
#helpers.get_top_features(training_data)

#classifier = GaussianNB()
classifier = RandomForestClassifier(n_estimators=200)
#classifier = SVC(kernel='rbf', C=100.0, gamma=0.1, random_state=0)
#classifier = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto')
#classifier = BernoulliNB()
#helpers.cross_validate(training_data, classifier)

#classifier = classifier.fit( training_data[0::, 1::], training_data[0::, 0] )
#predictions = classifier.predict(test_data).astype(int)
#helpers.write_predictions(test_ids, predictions, "predictions.csv")

def get_function_options():
    return filter(lambda x: not "__" in x, dir(c))
