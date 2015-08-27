from flask import Flask
from flask import render_template, url_for
import pandas as pd
import pandas as pd
import numpy as np
from flask_bootstrap import Bootstrap
from sklearn.linear_model import LinearRegression
from flask import request, redirect


app = Flask(__name__)
#from functions import recommender 

@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'] )
def predict():


	if request.method == 'GET':
		return redirect(url_for('hello'))


	filedata = request.files['file']

	if request.form['norm']== 'Yes':
		normalized = True
	else:
		normalized = False

	if request.form['inter'] =='Yes':
		intercept = True
	else:
		intercept = False

	independent = str(request.form['output'])

	lm = LinearRegression(fit_intercept = intercept, normalize = normalized)

	
	data = pd.read_csv(filedata)
	y = data.pop(independent).values
	X = data.values


	lm.fit(X,y)
	R2 = round(lm.score(X,y),3)
	coefs = lm.coef_

	features = data.columns.values

	coef_dict = {feature: round(coef,3) for (feature,coef) in zip(features,coefs)}


	return render_template("predict.html", coefs = coef_dict, r2 = R2)



if __name__ == '__main__':
	Bootstrap(app)
	app.run(host='0.0.0.0', port=8080, debug=True)


