import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## import ridge regressor model and standard scaler pickle

rodge_model =pickle.load(open("Model/ridge.pkl","rb"))
Standard_Scaler =pickle.load(open("Model/scaler.pkl","rb"))

## Route For Home Page
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        pass
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,BUI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template("home.html")
        




if __name__=="__main__":
    app.run(host="0.0.0.0")



    from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    # Run the git status command to check the status of the working directory
    git_status_output = subprocess.check_output(['git', 'status']).decode('utf-8')
    
    # Pass the git status information to the HTML template
    return render_template('index.html', git_status=git_status_output)

if __name__ == '__main__':
    app.run(debug=True)

