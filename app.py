from flask import Flask,request, render_template
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Initalise the Flask app
app = Flask(__name__,template_folder='Templates')
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods = ['POST'])
def classify():

    # features = ['Area', 'AspectRation', 'Eccentricity', 'Extent', 'Solidity',
    #    'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
    #    'ShapeFactor4']
    mean_list = [53059.189551140546, 1.583488653697225, 0.7510273459611002, 0.74973626312297, 0.9871449927863124, 0.8732599032822813, 0.7997713921236528, 0.006562861306198529, 0.001715068272500925, 0.995063369143438]
    std_list = [29319.105793355648, 0.24663455824767253, 0.09189423963556599, 0.04909250242013269, 0.004658886983197752, 0.05953038302435475, 0.06166245749381589, 0.0011269139788185635, 0.0005951618324030178, 0.004365831961602814]
    features_values = []
    stdscaled_fv = []
    for x in request.form.values() :
        features_values.append(x)
    # StandardScaler z = (x - u) / s
    for (value, mean, std) in zip(features_values, mean_list,std_list):
        z = (float(value) - mean)/std
        stdscaled_fv.append(z)
    final_features_values = np.array(stdscaled_fv, dtype=float).reshape(1,10)

    classification = model.predict(final_features_values)

    output = round(classification[0], 2)
    # if request.method == "POST":
    #     name = request.form["username"]

    match output:
        case 1 :
            output = 'SEKER'
        case 2 :
            output = 'BARBUNYA'
        case 3 :
            output = 'BOMBAY'
        case 4 :
            output = 'CALI'
        case 5 :
            output = 'DERMASON'
        case 6 :
            output = 'HOROZ'
        case 7 :
            output = 'SIRA'
        case _ :
            output = 'Unknown Class'

    return render_template("index.html", prediction_text = 'Drybean belongs to class {}'.format(output))

if(__name__ == "__main__"):
    app.run(debug=True)