#!/usr/bin/python

dots = "\n............"

def get_poi_count(dict):
    """
        It returns the count of POI on the dataset
    """
    count = 0
    for person in dict.keys():
        if dict[person]['poi'] == 1:
            count += 1
        else:
            continue
    return count

def get_suspicious_names(dict):
    """
        It returns a list of people that have a name with less than 2 words or more than 4 words.

        INPUT:
            dict: dict. The names must be the primary keys of the dict.
        RETURN:
            suspicious_names: 1Dimension list. It contains the names that have less than 2 words or more than 4 words.
    """
    suspicious_names = []
    for person in dict.keys():
        name_len = len(person.split(" "))
        if name_len < 2 or name_len > 4:
            suspicious_names.append(person)

    return suspicious_names

def pop_keys(dict, keys_list):
    """
        It removes all the primary keys within keys_list from a dict.

        INPUT:
            dict: dict. The keys you want to pop must be primary keys of this dict.
            keys_list: 1Dimension list. A list of the keys you want to pop.
        RETURN:
            dict: dict. It's the new dict after popping the keys_list keys.
    """
    for key in keys_list:
        dict.pop(key)

    return dict

def get_missing_count(dict):
    features_missing_count = []
    for key in dict[list(dict.keys())[0]]:
        if key != 'poi':
            missing_count = [key, 0]
            for person in dict.keys():
                if dict[person][key] == 0 or dict[person][key] == 'NaN':
                    missing_count[1] += 1
            features_missing_count.append(missing_count)
    return features_missing_count

def plot_vertical_bars(ind, y, width, title, xlabel, ylabel, xticks, color, rotation='horizontal', show=False):
    import matplotlib.pyplot as plt

    plt.bar(ind, y, width=width, color=color)

    for a, b in zip(ind, y):
        plt.text(width/2 + a, b + 1, b, ha= 'center', va='bottom', color='black')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ind + width/2, xticks, rotation=rotation, fontsize='small')
    plt.tight_layout()
    if show:
        plt.show()


def get_negative_count_people(dict):
    """
        It counts the negative values of the secondary keys of each primary key.
        Then it returns a dict where each primary dict key is a feature (secondary key from the input dict), its value is another dict with the following keys:
            'count': The number of rows of the feature that has got a negative value;
            'people': A list with the name of the people who's got a negative value on the respective feature.
        Returned dict format example:
        {
            'feature1': {
                'count': 3,
                'people': ['Joao', 'Joaquina', 'Betania']
            }
            'feature2': {
                'count': 98,
                'people': ['Abigail, 'Josefina']
            }
        }
    """
    negative = {}
    for person in dict.keys():
        for k, v in dict[person].items():
            if v < 0:
                if k in negative:
                    negative[k]['count'] += 1
                    negative[k]['people'].append(person)
                else:
                    negative[k] = {
                        'count': 1,
                        'people': [person]}

    return negative


def change_nan(dict, value=0):
    """
        It changes NaN values from dict by value.

        INPUT:
            dict: dict. The NaNs you want to replace must be values of secondary keys.
            value: The value that will replace the NaN.
        RETURN:
            dict: dict. The new dict with the replaced values.
    """
    for person in dict.keys():
        for k, v in dict[person].items():
            if v == 'NaN':
                dict[person][k] = value

    return dict

def change_negative_to_absolute(dict):
    """
        It changes NaN values from dict by its absolute values.

        INPUT:
            dict: dict. The negative values you want to replace must be values of secondary keys.
        RETURN:
            dict: dict. The new dict with only absolute values.
    """
    for person in dict.keys():
        for k, v in dict[person].items():
            if v < 0:
                dict[person][k] = abs(v)
    return dict

def get_almost_null_keys(dict, threshold):
    """
        It returns a list with the dict keys whose null features ratio is greather than the threshold.
    """
    import features
    almost_null_keys = []
    for person in dict.keys():
        count = 0
        for k, v in dict[person].items():
            if k == 'poi':
                continue
            else:
                if v == 'NaN' or v == 0:
                    count += 1
        null_features_ratio = float(count) / float(len(features.original_features))
        if null_features_ratio > threshold:
            almost_null_keys.append(person)

    return almost_null_keys

def create_new_ratio_feature(dict, ft_list, numerator, denominator):
    """
        It takes a numerator value and a denominator value, then divides one another to get the ratio.
        The ratio is add as a new key. It's name is = '<numerator_name> + _ratio'
        Thew new feature name is add to ft_list
    """
    for person in dict.keys():
        new_feature_name = numerator + '_ratio'
        num = dict[person][numerator]
        den = dict[person][denominator]
        if float(den) != 0.:
            ratio = float(num) / float(den)
        else:
            ratio = 0.
        dict[person][new_feature_name] = ratio
    ft_list.append(new_feature_name)

def create_new_squared_feature(dict, ft_list, to_square_feature):
    """
        It takes a value, then squares it.
        The squared value is add as a new key. It's name is = 'squared_ + <to_square_feature_name>'
        Thew new feature name is add to ft_list
    """
    for person in dict.keys():
        new_feature_name = 'squared_' + to_square_feature
        value = dict[person][to_square_feature]
        if float(value) != 0.:
            squared = float(value) * float(value)
        else:
            squared = 0.
        dict[person][new_feature_name] = squared
    ft_list.append(new_feature_name)

def apply_scaler(features):
    """
        It applies MinMaxScaler on a list of features and returns the scaled features
    """
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    features_array = np.array(list(features))
    scaler = MinMaxScaler()
    new_features_array = scaler.fit_transform(features_array)
    return new_features_array

def apply_kbest(features, labels, k, ft_list):
    """
        It applies KBest feature selection algorithm, prints the chosen features and chi2 scores, then returns the selected features
    """
    print dots + "Applying KBest. K =", k
    from sklearn.feature_selection import SelectKBest, chi2
    kbest = SelectKBest(chi2, k)
    selected_features = kbest.fit_transform(features, labels)
    scores = kbest.scores_
    ft_scores = zip(ft_list[1:], scores)

    print '\n\t{:<25}   {:>3}'.format('CHOSEN_FEATURE', 'CHI2_SCORE')
    count = 0
    for ft, score in sorted(ft_scores, key=lambda x: x[1], reverse=True):
        if count < k:
            print '\t{:<25}   {:>3}'.format(ft, score)
            count += 1
    return selected_features

def apply_classifier(classifier, X_train, y_train):
    """
        It fits the classifier on a X_train and y_train, prints the fitting time, then returns the classifier
    """
    import time
    t0 = time.time()
    classifier.fit(X_train, y_train)
    t1 = time.time()
    print "\t* Fitting time:", round(t1 - t0, 3), "s"

    return classifier

def print_scores(classifier, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import time
    t0 = time.time()
    pred = classifier.predict(X_test)
    t1 = time.time()
    print "\t* Prediction time:", round(t1 - t0, 3), "s"
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print '\n\t{:<10}   {:<10}   {:<10}   {:<10}'.format('Acuracy', 'Precision', 'Recall', 'F1')
    print '\t{:<10}   {:<10}   {:<10}   {:<10}'.format(accuracy, precision, recall, f1)

def feature_format(dataset, features_list):
    from feature_format import featureFormat, targetFeatureSplit
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    return features, labels

def dataset_split(features, labels, test_size):
    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=test_size, random_state=42)

    return features_train, features_test, labels_train, labels_test

def pretreatment(dataset, features_list, scaling=True, selection=10, test_size=.3):
    features, labels = feature_format(dataset, features_list)
    if scaling:
        features = apply_scaler(features)
    if selection:
        features = apply_kbest(features, labels, selection, features_list)
    features_train, features_test, labels_train, labels_test = dataset_split(features, labels, test_size)

    return features_train, features_test, labels_train, labels_test

def apply_gridsearchcv(classifier, X, y, param_grid, scoring, refit, cv, verbose=False):
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(classifier, param_grid=param_grid, scoring=scoring, cv=cv, refit=refit, verbose=verbose, return_train_score=False)
    gs.fit(X, y)

    return gs.best_estimator_

def get_feature_importance(features_list, importances_list):
    features_list = features_list[1:]
    ft_importance_list = []
    for x in zip(features_list, importances_list):
        ft_importance_list.append(list(x))
    ft_importance_list.sort(reverse=True, key=lambda x:x[1])

    return ft_importance_list

def print_feature_by_importance(ft_importance_list):
    feature_list, importance_list = zip(*ft_importance_list)
    print '['
    for i in range(len(feature_list)):
        if i != (len(feature_list) - 1):
            if i == 0:
                print "    'poi',"
            print "    '" + feature_list[i] + "',"
        else:
            print "    '" + feature_list[i] + "'"
            print ']'
