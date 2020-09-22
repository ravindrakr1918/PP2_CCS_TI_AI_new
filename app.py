

from flask import Flask, render_template, request
import pandas as pd
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
    return render_template('index1.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(X) for X in request.form.values()]
    
    print(int_features)
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=prediction[0]
    CCS=round(output[0],2)
    TI=round(output[1],2)
    AI=round(output[2],2)
    #ravi = pd.DataFrame(prediction, columns=['CCS', 'TI', 'AI']).to_csv('prediction.csv')
    return render_template('index1.html', prediction_text='CCS(kg/P) is {}'.format(CCS), prediction_TI='TI(+6.3 mm)% is {}'.format(TI), prediction_AI='AI(-0.5 mm)% is {}'.format(AI), 
    input_1=int_features[0],input_2=int_features[1],input_3=int_features[2],input_4=int_features[3],input_5=int_features[4],input_6=int_features[5],input_7=int_features[6],input_8=int_features[7],input_9=int_features[8],input_10=int_features[9],input_11=int_features[10],input_12=int_features[11],input_13=int_features[12],input_14=int_features[13],input_15=int_features[14],input_16=int_features[15],input_17=int_features[16],input_18=int_features[17],input_19=int_features[18],input_20=int_features[19],input_21=int_features[20],input_22=int_features[21],input_23=int_features[22],input_24=int_features[23])
if __name__=="__main__":
    app.run(debug=True)