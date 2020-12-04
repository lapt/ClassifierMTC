# app.py
from flask import Flask
from flask_restful import Api, Resource, reqparse
from sklearn.externals import joblib
import numpy as np

APP = Flask(__name__)
API = Api(APP)

IRIS_MODEL = joblib.load('results_models_test/LinerSVC.joblib')


class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('asunto')
        args = parser.parse_args()  # creates dict

        X_new = np.fromiter(args.values(), dtype='S128')  # convert input to array

        out = {'Prediction': int(IRIS_MODEL.predict([X_new[0]])[0])}

        return out, 200


API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True, port='1080')