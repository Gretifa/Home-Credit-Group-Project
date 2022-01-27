from flask import Flask, request
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gc
import pickle
import sqlite3
import flask

app = Flask(__name__)

with open('model/model.pkl', 'rb') as f:
	test_predictor_class = pickle.load(f)
	
#home page
@app.route('/', methods = ['GET'])
def home_page():
    return flask.render_template('home-page.html')

#prediction page
@app.route('/home', methods = ['POST', 'GET'])
def inputs_page():
	return flask.render_template('predict.html')

#results page
@app.route('/predict', methods = ['POST'])
def prediction():
	conn = sqlite3.connect('model/data.db')
    #getting the SK_ID_CURR from user
	sk_id_curr = request.form.to_dict()['SK_ID_CURR']
	sk_id_curr = int(sk_id_curr)
	test_datapoint = pd.read_sql_query(f'SELECT * FROM applicators WHERE SK_ID_CURR == {sk_id_curr}', conn)
	test_datapoint = test_datapoint.drop(["index",],axis=1)
	print(test_datapoint)
	test_datapoint = test_datapoint.replace([None], np.nan)
	predicted_class= test_predictor_class.predict(test_datapoint)
	predicted_proba = test_predictor_class.predict_proba(test_datapoint)[:, 1]
	predicted_proba = round(predicted_proba[0] * 100 ,2)
	print(predicted_proba)
	if predicted_class == 1:
		prediction = 'a Potential Defaulter'
	else:
		prediction = 'not a Defaulter'
		predicted_proba = round(100 - predicted_proba,2)


	conn.close()
	gc.collect()

	return flask.render_template('result_and_inference.html', output_proba = predicted_proba,output_class = prediction, sk_id_curr = sk_id_curr)

if __name__ == '__main__':
	app.run(host = '0.0.0.0', port = 5000)