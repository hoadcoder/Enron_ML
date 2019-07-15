#!/usr/bin/python

import features, functions
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

dots = "\n............"

# Import all the original features excluding 'email_address'.
features_list = features.original_features

# These variables control whether or not the non-tuned classifiers and/or the tuned classifier will run.
# The tuned classifier is the one that will be exported to be tested on tester.py
run_non_tuned_classifiers = False
run_tuned_classifier = True

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

original_dataset_size = len(data_dict)
original_poi_count = functions.get_poi_count(data_dict)
original_non_poi_count = original_dataset_size - original_poi_count

features_missing_count_0 = functions.get_missing_count(data_dict)

print dots + "Deleting 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' from the dataset"
suspicious_names = functions.get_suspicious_names(data_dict)
# suspicious_names = ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL']
data_dict = functions.pop_keys(data_dict, suspicious_names)

threshold = .85
print dots + "Deleting people which {}% of keys are null".format(threshold*100)
almost_null_keys = functions.get_almost_null_keys(data_dict, threshold)
data_dict = functions.pop_keys(data_dict, almost_null_keys)

negative_dict = functions.get_negative_count_people(data_dict)

print dots + "Printing no. of Negative values"
print '\n\t{:<25}   {:>3}'.format('FEATURE', 'COUNT')
for k, v in negative_dict.items():
    print '\t{:<25}   {:>3}'.format(k, v['count'])

strange_negative_count_features = ['restricted_stock', 'deferral_payments', 'total_stock_value']
printed_people = []
for feature in strange_negative_count_features:
    people = negative_dict[feature]['people']
    for person in people:
        if person not in printed_people:
            print dots + person
            for k, v in data_dict[person].items():
                print '\t{:<25}   {:>3}'.format(k, v)
            printed_people.append(person)

print dots + "Correcting 'BELFER ROBERT' and 'BHATNAGAR SANJAY' values according to 'enron61702insiderpay.pdf' file"
data_dict['BELFER ROBERT']['deferred_income'] = 102500
data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['exercised_stock_options'] = 'NaN'
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = 44093
data_dict['BELFER ROBERT']['total_stock_value'] = 'NaN'

data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = 2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

print dots + "Changing NaN values by Zeroes"
data_dict = functions.change_nan(data_dict)

print dots + "Changing Negative values by Positive values"
data_dict = functions.change_negative_to_absolute(data_dict)

clean_dataset_size = len(data_dict)
clean_poi_count = functions.get_poi_count(data_dict)
clean_non_poi_count = clean_dataset_size - clean_poi_count
print "\nOriginal Dataset Size: {}\tClean Dataset Size: {}".format(original_dataset_size, clean_dataset_size)
print "Original POI Count: {}\t\tClean POI Count: {}".format(original_poi_count, clean_poi_count)
print "Original non-POI Count: {}\tClean non-POI Count: {}".format(original_non_poi_count, clean_non_poi_count)

features_missing_count_1 = functions.get_missing_count(data_dict)
title = 'Count of missing values by features before and after cleaning'
width = .85
label = 'Before Cleaning'
features_missing_count_0.sort(reverse=True, key= lambda x:x[1])
x_0, y_0 = zip(*features_missing_count_0)
ind_0 = np.arange(start=0, step=3, stop=3*len(x_0))
plt.bar(ind_0, y_0, width=width, color='#3E6B6E', label=label)

label = 'After Cleaning'
features_missing_count_1.sort(reverse=True, key= lambda x:x[1])
x_1, y_1 = zip(*features_missing_count_1)
ind_1 = np.arange(start=1, step=3, stop=1 + 3*len(x_1))

plt.bar(ind_1, y_1, width=width, color='#6AB592', label=label)

for a, b in zip(ind_0, y_0):
    plt.text(width/2 + a, b + 1, b, ha= 'center', va='bottom', color='black')

for a, b in zip(ind_1, y_1):
    plt.text(width/2 + a, b + 1, b, ha= 'center', va='bottom', color='black')

plt.title(title)
plt.xlabel('Features')
plt.ylabel('Count of missing values')
plt.xticks(ind_0 + 1, x_0, rotation='vertical', fontsize='medium')
plt.tight_layout()
plt.legend()
plt.show()

# Store to my_dataset for easy export below.
my_dataset = data_dict

print dots + "Creating a bunch of new ratio and squared features"
new_features_list = features_list[:]

functions.create_new_ratio_feature(my_dataset, new_features_list, 'bonus', 'total_payments')
functions.create_new_ratio_feature(my_dataset, new_features_list, 'salary', 'total_payments')
functions.create_new_ratio_feature(my_dataset, new_features_list, 'expenses', 'total_payments')
functions.create_new_ratio_feature(my_dataset, new_features_list, 'from_poi_to_this_person', 'from_messages')
functions.create_new_ratio_feature(my_dataset, new_features_list, 'from_this_person_to_poi', 'to_messages')
functions.create_new_ratio_feature(my_dataset, new_features_list, 'exercised_stock_options', 'total_stock_value')
functions.create_new_ratio_feature(my_dataset, new_features_list, 'restricted_stock', 'total_stock_value')

functions.create_new_squared_feature(my_dataset, new_features_list, 'salary')
functions.create_new_squared_feature(my_dataset, new_features_list, 'bonus')
functions.create_new_squared_feature(my_dataset, new_features_list, 'expenses')
functions.create_new_squared_feature(my_dataset, new_features_list, 'salary_ratio')
functions.create_new_squared_feature(my_dataset, new_features_list, 'bonus_ratio')
functions.create_new_squared_feature(my_dataset, new_features_list, 'expenses_ratio')
functions.create_new_squared_feature(my_dataset, new_features_list, 'from_poi_to_this_person_ratio')
functions.create_new_squared_feature(my_dataset, new_features_list, 'from_this_person_to_poi_ratio')
functions.create_new_squared_feature(my_dataset, new_features_list, 'exercised_stock_options_ratio')
functions.create_new_squared_feature(my_dataset, new_features_list, 'restricted_stock_ratio')


if run_non_tuned_classifiers:
    print dots + "Running Non-Tuned Classifiers"
    features_train, features_test, labels_train, labels_test = \
        functions.pretreatment(my_dataset, new_features_list, scaling=True, selection=False, test_size=.3)

    print dots + "Naive Bayes:"
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()
    classifier = functions.apply_classifier(nb, features_train, labels_train)
    # test_classifier(classifier, my_dataset, new_features_list)

    print dots + "Support Vector Machines:"
    from sklearn.svm import SVC
    svm = SVC(kernel='linear', max_iter=1000)
    classifier = functions.apply_classifier(svm, features_train, labels_train)
    # feature_importance_list = functions.get_feature_importance(new_features_list, classifier.coef_[0])
    # functions.print_feature_by_importance(feature_importance_list)
    # test_classifier(classifier, my_dataset, new_features_list)

    print dots + "K Nearest Neighbors:"
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    classifier = functions.apply_classifier(knn, features_train, labels_train)
    # test_classifier(classifier, my_dataset, new_features_list)

    print dots + "Decision Tree:"
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    classifier = functions.apply_classifier(dt, features_train, labels_train)
    # feature_importance_list = functions.get_feature_importance(new_features_list, classifier.feature_importances_)
    # functions.print_feature_by_importance(feature_importance_list)
    # test_classifier(classifier, my_dataset, new_features_list)

if run_tuned_classifier:

    # If run_final_classifier is False, it'll run the classifier a few time for a different number of selected features
    # If run_final_classifier is True, it'll run the tuned clasifier to export
    run_final_classifier = True

    features_list = new_features_list

    # When the below line is not a comment, the classifier will only use the default features, ignoring the created ones
    #features_list = features.original_features

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
    }
    scoring = ['precision', 'recall']

    if run_final_classifier == False:
        k_list = range(3, len(features_list))
        for k in k_list:
            print "#################################################################"
            features, labels = functions.feature_format(my_dataset, features_list)

            features = functions.apply_scaler(features)
            features = functions.apply_kbest(features, labels, k, features_list)

            classifier = functions.apply_gridsearchcv(knn, features, labels, param_grid=param_grid, scoring=scoring, refit='recall', cv=3, verbose=False)
            test_classifier(classifier, my_dataset, features_list)
        sys.exit()
    else:
        features, labels = functions.feature_format(my_dataset, features_list)

        features = functions.apply_scaler(features)
        features = functions.apply_kbest(features, labels, 4, features_list)

        print dots + "Tuned K Nearest Neighbors:"

        classifier = functions.apply_gridsearchcv(knn, features, labels, param_grid=param_grid, scoring=scoring, refit='recall', cv=3, verbose=False)
        test_classifier(classifier, my_dataset, features_list)
        clf = classifier

# Example starting point. Try investigating other evaluation techniques!
    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

    dump_classifier_and_data(clf, my_dataset, features_list)