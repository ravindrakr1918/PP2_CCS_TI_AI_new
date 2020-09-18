

from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('PP2_CCS_TI_AI3.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(X) for X in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=prediction[0]
    CCS=round(output[0],2)
    TI=round(output[1],2)
    AI=round(output[2],2)
    return render_template('index.html', prediction_text='CCS(kg/P) is {}'.format(CCS), prediction_TI='TI(+6.3 mm)% is {}'.format(TI), prediction_AI='AI(-0.5 mm)% is {}'.format(AI))


if __name__=="__main__":
    app.run(debug=True)