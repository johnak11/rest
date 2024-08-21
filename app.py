import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
salary_deploy = pickle.load(open('startup_prediction.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])

def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = salary_deploy.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html',prediction_text ='Profit for the startup should be $ {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = salary_deploy.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
