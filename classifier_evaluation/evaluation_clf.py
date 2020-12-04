# -*- coding: utf-8 -*-

import pandas
import csv
from data_processing.Clean_data import clean, tokenize, spanish_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
import joblib
from classifier_evaluation.EstimatorSelectionHelper import EstimatorSelectionHelper
from typing import Sized


def create_subject_corpus(path_archive):
    subject_corpus = pandas.read_csv(path_archive, sep='\t', quoting=csv.QUOTE_NONE,
                                   names=["idSubject", "subject", "priority"])
    subject_corpus['subject'] = subject_corpus['subject'].map(lambda text: clean(text))
    # print subject_corpus.groupby('priority').describe()
    # print subject_corpus
    return subject_corpus


def create_classifier(classifier, scoring, params, subject_corpus, path_save_cls):
    msg_train, msg_test, label_train, label_test = train_test_split(subject_corpus['subject'], subject_corpus['priority'],
                                                                    test_size=0.2, stratify=subject_corpus['priority'])

    count_vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=tokenize,
        lowercase=True,
        stop_words=spanish_stopwords,
        decode_error='ignore'
    )

    pipeline = Pipeline([
        ('vect', count_vectorizer),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('cls', classifier)  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    grid = GridSearchCV(
        pipeline,  # pipeline from above
        params,  # parameters to tune via cross validation
        refit=True,  # fit using all available data at the end, on the best found param combination
        n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
        scoring=scoring,  # what score are we optimizing?
        cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
    )

    nb_detector = grid.fit(msg_train, label_train)
    # print nb_detector.grid_scores_
    print(nb_detector.best_estimator_)
    print(grid.best_params_)
    print(grid.best_score_)
    print(grid.scoring)
    # predictions = nb_detector.predict(msg_test)
    # print confusion_matrix(label_test, predictions)
    # print classification_report(label_test, predictions)
    joblib.dump(grid.best_estimator_, path_save_cls)
    return grid.scoring


def max_classification_random(path_corpus, path_result,
                              path_result_cls_transport, part_data):  # , path_cls_lineal, path_cls_svm, path_cls_forest, path_cls_bayes):
    subject_corpus = create_subject_corpus(path_corpus)
    post = subject_corpus[subject_corpus['priority'] == 1]
    neg = subject_corpus[subject_corpus['priority'] == 0]

    post = post[:part_data]
    neg = neg[:part_data]

    print("Number positive Priority: %d" % len(post))
    print("Number negative Priority: %d" % len(neg))

    subject_corpus = pandas.concat([post, neg], ignore_index=True)

    msg_train = subject_corpus['subject']
    label_train = subject_corpus['priority']

    scoring = 'roc_auc'

    parameters_random = {'tfidf__use_idf': (True, False),
                         'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas or bigramas
                         "cls__max_depth": [5, None],
                         "cls__max_features": [None, 10, "auto", "sqrt", "log2"],
                         # "cls__min_samples_split": [1.0, 2, 3, 10],
                         # "cls__min_samples_leaf": [1, 3, 10],
                         # "cls__bootstrap": [True, False],
                         "cls__criterion": ["gini", "entropy"],
                         "cls__n_estimators": [40, 80, 100, 150]}

    parameters_tree = {'tfidf__use_idf': (True, False),
                       'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas or bigramas
                       "cls__max_depth": [5, None],
                       "cls__max_features": [None, 10, "auto", "sqrt", "log2"],
                       "cls__criterion": ["gini", "entropy"]
                       }

    parameters_linear = {'tfidf__use_idf': (True, False),
                         'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas or bigramas
                         'cls__C': (0.2, 0.5, 0.7, 0.8, 1.0, 2., 10.0, 100.0),
                         'cls__loss': ('hinge', 'squared_hinge')
                         }

    parameters_multinomial = {'tfidf__use_idf': (True, False),
                              'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas or bigramas
                              'cls__alpha': (1.0, 1e-2, 1e-3)}

    parameters_svc = {'tfidf__use_idf': (True, False),
                      'vect__ngram_range': ((1, 1), (1, 2)),  # unigramas or bigramas
                      'cls__C': (0.2, 0.5, 0.7, 0.8, 1.0, 2., 10.0, 100.0),
                      'cls__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    # MODELS
    params1 = {
        'RandomForestClassifier': parameters_random,
        'treeClassifier': parameters_tree,
        'LinearSVC': parameters_linear,
        'MultinomialNB': parameters_multinomial,
        'SVC': parameters_svc
    }

    models1 = {
        'treeClassifier': tree.DecisionTreeClassifier(),
        'LinearSVC': svm.LinearSVC(),
        'MultinomialNB': MultinomialNB(),
        'SVC': svm.SVC(),
        'RandomForestClassifier': RandomForestClassifier()
    }

    helper1 = EstimatorSelectionHelper(models1, params1)
    helper1.fit(msg_train, label_train, scoring=scoring, n_jobs=-1, refit=True,
                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
                path_result_cls_transport=path_result_cls_transport)
    df_result = helper1.score_summary(sort_by='mean_score')
    print(df_result.head())
    df_result.to_csv(path_result, index=False, header=True, encoding='utf-8', sep="\t")
