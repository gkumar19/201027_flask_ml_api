# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:57:47 2020

@author: Gaurav
"""

from flask import Flask, jsonify, request
import joblib
import pandas as pd
import traceback

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    try:
        recieved_json = request.json
        query_df = pd.DataFrame(recieved_json, columns=['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked'])
        prediction = {"prediction": model.predict(query_df).tolist()}
        return jsonify(prediction)
    except:
        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    app.run(debug=True)

#For Postman app use following in url,  http://127.0.0.1:5000/prediction, and give either GET or POST request using JSON file
#Example json input:
#[{"PassengerId":1, "Pclass":3, "Sex":"male", "Age":22, "SibSp":1, "Parch":0, "Fare":7.25, "Embarked":"S"},
#{"PassengerId":2, "Pclass":1, "Sex":"female", "Age":38, "SibSp":1, "Parch":0, "Fare":71.2833, "Embarked":"C"}]    

#For Heroku Deployment
#Generate requirement.txt using conda list --export > requirements.txt
#In requirement.txt Add gunicorn==20.0.4