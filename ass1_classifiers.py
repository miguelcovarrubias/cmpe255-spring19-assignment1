import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score


def load_labeled_words(file_name):
    data = {}
    with open(file_name) as f:
        for line in f:
            sp = line.split()
            data[' '.join(sp[:-1])] = sp[-1]
    return data


def load_and_label_reviews(file_name, labeled_data):
    labels = []
    data = []
    stars_indexes = []
    with open(file_name) as f:
        for line in f:
            review_text = str(json.loads(line)["text"]).lower().replace("\n", " ")

            global_sum = 0
            for word in review_text.split():
                sum_ = 0
                cnt = 0
                for k in labeled_data:
                    if k in word:
                        sum_ = sum_ + int(labeled_data[k])
                        cnt = cnt + 1
                global_sum = global_sum + sum_

            if global_sum > 3:
                label = "positive"
            elif global_sum < -3:
                label = "negative"
            else:
                label = "neutral"

            stars_indexes.append(str(json.loads(line)["stars"]))

            labels.append(label)
            data.append(review_text)
    return data, labels, stars_indexes

def transform_to_features(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer='word',
        lowercase=False,
    )
    features = vectorizer.fit_transform(
        data
    )
    features_nd = features.toarray()
    return features_nd

def knn_clf_fun(X_train, X_test, y_train, y_test):
    start_time = time.time()

    knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=10)
    knn_clf.fit(X_train, y_train)

    y_knn_pred = knn_clf.predict(X_test)

    acc_sc = accuracy_score(y_test, y_knn_pred)
    print("--- KNN got accuracy of %s and it took %s min ---" % (acc_sc, ((time.time() - start_time) / 60)))


def sgd_clf_fun(X_train, X_test, y_train, y_test):

    print("Starting the SGDClassifier with max iterations of 100 and using all test data")

    sgd_clf = SGDClassifier(max_iter=100, tol=-np.infty, random_state=1234)
    sgd_clf.fit(X_train, y_train)

    y_train_pred = sgd_clf.predict(X_test)

    acc_sc = accuracy_score(y_test, y_train_pred)
    print("--- SGD got accuracy score of %s ---" % acc_sc)

    cross_val_sc = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
    print("--- SGD got cross validation score of %s ---" % cross_val_sc)

    pres_sc = precision_score(y_test, y_train_pred, average='macro')
    print("--- SGD got precision score of %s ---" % pres_sc)

    rec_sc = recall_score(y_test, y_train_pred, average='macro')
    print("--- SGD got recall score of %s ---" % rec_sc)

    f1_sc = f1_score(y_test, y_train_pred, average='macro')
    print("--- SGD got F1 score of %s ---" % f1_sc)


def sgd_clf_fun_five_stars(X_train, X_test, y_train, y_test, five_star_indices, features_nd, data):

    print("Starting the SGDClassifier with max iterations of 100 and using only the test data associated with 5 stars")

    sgd_clf = SGDClassifier(max_iter=100, tol=-np.infty, random_state=1234)
    sgd_clf.fit(X_train, y_train)

    # remove add indexes of X_test that are not 5.0
    X_test_five_stars = []
    y_test_five_stars = []
    for ind in five_star_indices:
        X_test_five_stars.append(X_test[ind])
        y_test_five_stars.append('positive')

    y_train_pred = sgd_clf.predict(X_test_five_stars)

    positive_predictions = []
    negative_predictions = []
    neutral_predictions = []

    if len(five_star_indices) > 400:
        sample_size = 400
    else:
        sample_size = len(five_star_indices)

    for val in range(0, sample_size):
        i = five_star_indices[val]
        index_ = features_nd.tolist().index(X_test[i].tolist())
        #print("::{}::{}".format(y_train_pred[i], data[index_]))
        if str(y_train_pred[val]) == 'positive':
            positive_predictions.append(data[index_])
        elif str(y_train_pred[val]) == 'negative':
            negative_predictions.append(data[index_])
        else:
            neutral_predictions.append(data[index_])

    print("Few instances where it predicted positive: ")
    for i in range(0, 5):
        if len(positive_predictions) > i:
            print("-->", positive_predictions[i])
        else:
            break

    print()

    print("Few instances where it predicted negative but it should be positive: ")
    for i in range(0, 5):
        if len(negative_predictions) > i:
            print("-->", negative_predictions[i])
        else:
            break

    print()

    print("Few instances where it predicted neutral but it should be positive: ")
    for i in range(0, 5):
        if len(neutral_predictions) > i:
            print("-->", neutral_predictions[i])
        else:
            break

    print()

    acc_sc = accuracy_score(y_test_five_stars, y_train_pred)
    print("--- SGD got accuracy score of %s ---" % acc_sc)

    cross_val_sc = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
    print("--- SGD got cross validation score of %s ---" % cross_val_sc)

    pres_sc = precision_score(y_test_five_stars, y_train_pred, average='macro')
    print("--- SGD got precision score of %s ---" % pres_sc)

    rec_sc = recall_score(y_test_five_stars, y_train_pred, average='macro')
    print("--- SGD got recall score of %s ---" % rec_sc)

    f1_sc = f1_score(y_test_five_stars, y_train_pred, average='macro')
    print("--- SGD got F1 score of %s ---" % f1_sc)


if __name__ == "__main__":

    # modify the following paths appropriately
    labeled_words_file_path = "data/labeled_words.txt"
    yelp_review_data_path = "data/review_20000.json"

    # read start labeling the data by using labeled_words_file
    labeled_data = load_labeled_words(labeled_words_file_path)

    # load and label the reviews from the dataset
    data, data_labels, stars_indexes = load_and_label_reviews(yelp_review_data_path, labeled_data)

    print("Labeling data stats:")
    print("Positive entries: %s" % data_labels.count("positive"))
    print("Negative entries: %s" % data_labels.count("negative"))
    print("Neutral entries: %s" % data_labels.count("neutral"))

    # transform data into a matrix
    features_nd = transform_to_features(data)

    # split data 80% for testing and 20% for testing (make sure to use random_state=1234)

    X_train, X_test, y_train, y_test = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80,
        random_state=1234
    )

    # Start the SGD classifier function
    sgd_clf_fun(X_train, X_test, y_train, y_test)

    # split data 80% for testing and 20% for testing (make sure to use random_state=1234)
    # do this again to keep track of the stars list indexes
    _, _, y_train_ratings, y_test_ratings = train_test_split(
        features_nd,
        stars_indexes,
        train_size=0.80,
        random_state=1234)

    # record the indices associated with 5.0 start ratings
    five_star_indices = [i for i, x in enumerate(y_test_ratings) if x == '5.0']

    sgd_clf_fun_five_stars(X_train, X_test, y_train, y_test, five_star_indices, features_nd, data)

    # knn clf
    knn_clf_fun(X_train, X_test, y_train, y_test)






