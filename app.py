# app.py
from flask import Flask
from flask_restful import Api, Resource, reqparse
from sklearn.externals import joblib
import numpy as np

APP = Flask(__name__)
API = Api(APP)

IRIS_MODEL = joblib.load('results_models_test/SVC.joblib')


class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('asunto')
        args = parser.parse_args()  # creates dict

        X_new = np.fromiter(args.values(), dtype='S128')  # convert input to array

        out = {'High_priority': int(IRIS_MODEL.predict([X_new[0]])[0]),
               'High_priority_prob': IRIS_MODEL.predict_proba([X_new[0]])[0][1]}

        return out, 200


API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True, port='1080')