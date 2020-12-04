import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from data_processing.Clean_data import *
from sklearn.pipeline import Pipeline
import joblib
import numpy as np


class EstimatorSelectionHelper(object):
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False, path_result_cls_transport="cls_name_1"):
        count_vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=tokenize,
            lowercase=True,
            stop_words=spanish_stopwords,
            decode_error='ignore'
        )

        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            pipeline = Pipeline([
                ('vect', count_vectorizer),  # strings to token integer counts
                ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
                ('cls', model)  # train on TF-IDF vectors w/ Naive Bayes classifier
            ])
            params = self.params[key]
            gs = GridSearchCV(pipeline, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X, y)
            joblib.dump(gs.best_estimator_, path_result_cls_transport)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].n_splits_):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
