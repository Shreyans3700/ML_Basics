import pickle

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## Import Prediction Model and Standardization Model
elasticnet_model = pickle.load(open('models/elasticnet_cv.pkl', 'rb'))
scaler_model = pickle.load(open('models/standardize.pkl', 'rb'))
 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictData', methods=['GET', 'POST'])
def predictDataPoint():
    if request.method == "POST":
        Tempreature = float(request.form.get('Tempreature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = int(request.form.get('Classes'))
        Region = int(request.form.get('Region'))    

        new_data_scaled = scaler_model.transform([[Tempreature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])  

        result = elasticnet_model.predict(new_data_scaled) 

        return render_template('home.html', results = result[0]) 
    else:
        return render_template("home.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0')