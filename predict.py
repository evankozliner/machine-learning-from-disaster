import helpers
import cleaners as c
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

# TODO can't clean with the exact same function because I'm dropping the worst features
cleaning_functions = [c.integer_sex_mapping, 
        c.clean_missing_embarks_with_mode,
        c.integer_embarked_mapping,
        c.clean_missing_ages_with_medians,
        c.clean_missing_fares_with_medians,
        c.drop_leftover_features]

train_ids, training_data = helpers.clean('data/train.csv', cleaning_functions)
test_ids, test_data = helpers.clean('data/test.csv', cleaning_functions)

features_map = c.get_top_features(training_data)
training_data = c.keep_n_top_features(3, features_map, training_data)
test_data = c.keep_n_top_features(3, features_map, test_data)

print training_data
#clf1 = LogisticRegression(random_state=1)
#clf2 = RandomForestClassifier(random_state=1)
#clf3 = GaussianNB()
#classifier = classifier.fit(X, y)

#clf0 = LogisticRegression()
clf1 = AdaBoostClassifier(n_estimators=1000)
#classifier = GaussianNB()
#clf4 = RandomForestClassifier(n_estimators=1000)
#classifier = SVC(kernel='rbf', C=100.0, gamma=0.1, random_state=0)
#clf2 = KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='auto')
#clf3 = BernoulliNB()
#classifier = VotingClassifier(estimators=[
    #('lr', clf0), ('ab', clf1), ('knn', clf2), ('bnb', clf3), ('rfc', clf4)], voting='hard')
helpers.cross_validate(training_data, classifier)

classifier = classifier.fit( training_data.values[0::, 1::], training_data.values[0::, 0] )
predictions = classifier.predict(test_data.values).astype(int)
helpers.write_predictions(test_ids, predictions, "predictions.csv")

def get_function_options():
    return filter(lambda x: not "__" in x, dir(c))
