from crypt import methods
from flask import Flask,render_template,request
import joblib
import pandas as pd
import numpy as np
from yaml import Mark

app= Flask(__name__)

mul_reg = open("multiple_linear_model.pkl", "rb")
ml_model = joblib.load(mul_reg)


@app.route('/')

def Home():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        print(request.form.get('NewYork'))
        try:
          NewYork=float(request.form['NewYork'])
          California=float(request.form['California'])
          Florida=float(request.form['Florida'])
          RnDSpend=float(request.form['RnDSpend'])
          AdminSpend=float(request.form['AdminSpend'])
          MarketSpend=float(request.form['MarketSpend'])
          pred_args=[NewYork,California,Florida,RnDSpend,AdminSpend,MarketSpend]
          pred_args_arr=np.array(pred_args)
          pred_args_arr=pred_args_arr.reshape(1,-1)
        #   mul_reg=open("multiple_linear_regression.pkl","rb")
        #   ml_model=joblib.load(mul_reg)
          model_prediction=ml_model.predict(pred_args_arr)
          model_prediction=round(float(model_prediction),2)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html',prediction=model_prediction)

if __name__=='__main__':
  app.run(host='0.0.0.0')