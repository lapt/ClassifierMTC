import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from data_processing.Clean_data import *
from sklearn.pipeline import Pipeline
import joblib
from sklearn import metrics
from classifier_evaluation.evaluation_clf import create_subject_corpus
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB


def test_model(model, X_train, y_train, X_test, y_test, ngram_range, use_idf):
    count_vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=tokenize,
        lowercase=True,
        stop_words=spanish_stopwords,
        decode_error='ignore',
        ngram_range=ngram_range
    )

    text_cls = Pipeline([
        ('vect', count_vectorizer),  # strings to token integer counts
        ('tfidf', TfidfTransformer(use_idf=use_idf)),  # integer counts to weighted TF-IDF scores
        ('cls', model)  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    text_cls.fit(X_train, y_train)
    predicted = text_cls.predict(X_test)
    result = metrics.classification_report(y_test, predicted)
    print(result)
    print(metrics.confusion_matrix(y_test, predicted))
    # df_result.to_csv(path_result, index=False, header=True, encoding='utf-8', sep="\t")
    return text_cls


def evaluation(path_corpus_train, path_corpus_test):
    subject_corpus_train = create_subject_corpus(path_corpus_train)
    msg_train = subject_corpus_train['subject']
    label_train = subject_corpus_train['priority']
    subject_corpus_test = create_subject_corpus(path_corpus_test)
    msg_test = subject_corpus_test['subject']
    label_test = subject_corpus_test['priority']
    print("RandonForest")
    model = RandomForestClassifier(criterion='gini', max_features='log2', n_estimators=150)
    model_predict = test_model(model, msg_train, label_train, msg_test, label_test, (1, 1), False)
    joblib.dump(model_predict, "results_models_test/RandonForest.joblib")

    print("LinerSVC")
    model = svm.LinearSVC(C=2, loss='squared_hinge')
    model_predict = test_model(model, msg_train, label_train, msg_test, label_test, (1, 2), True)
    joblib.dump(model_predict, "results_models_test/LinerSVC.joblib")

    print("MultinomialNB")
    model = MultinomialNB(alpha=0.01)
    model_predict = test_model(model, msg_train, label_train, msg_test, label_test, (1, 2), False)
    joblib.dump(model_predict, "results_models_test/MultinomialNB.joblib")

    print("SVC")
    model = svm.SVC(C=100, kernel='rbf')
    model_predict = test_model(model, msg_train, label_train, msg_test, label_test, (1, 2), False)
    joblib.dump(model_predict, "results_models_test/SVC.joblib")
