# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:25:09 2021

@author: abc
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('weight-height.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input = np.array([float(x) for x in request.form.values()]).reshape(-1,1)
    #print(input)
    output = model.predict(input)
    output = round(output[0],3)
    return render_template('index.html', prediction_text = 'Weight should be {}lbs.'.format(output))
    
    
if __name__ == "__main__":
    app.run(debug=True)