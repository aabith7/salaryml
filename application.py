from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pickle

application = Flask(__name__)
app = application

reg_model = pickle.load(open('models/deploy_regressor.pkl', 'rb'))
standard_scaler = pickle.load(open('models/deploy_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods = ['GET','POST'])
def model_prediction() :
    if request.method == 'POST':
        experiance = float(request.form.get('Experience'))
        education = float(request.form.get('Education_level'))
        age = float(request.form.get('age'))
        new_data_x = [experiance, education, age]
        new_data =standard_scaler.transform([new_data_x])
        result = reg_model.predict(new_data)
        return render_template('form.html',results =int(result))
    else:
        return render_template('form.html' )




if __name__ == '__main__':
    app.run(debug=True)