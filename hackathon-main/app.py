from flask import Flask, render_template, request
import numpy as np
import joblib  # To load the saved model and scaler

app = Flask(__name__)

# Load the saved model and preprocessing objects
kmeans = joblib.load('kmeans_model.joblib')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data from user input
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        job = int(request.form['job'])
        housing = int(request.form['housing'])
        saving_accounts = int(request.form['saving_accounts'])
        checking_account = int(request.form['checking_account'])
        purpose = int(request.form['purpose'])
        credit_amount = float(request.form['credit_amount'])
        duration = float(request.form['duration'])

        # Create the feature vector from user input
        input_data = np.array([[age, sex, job, housing, saving_accounts, checking_account, 
                                purpose, credit_amount, duration]])

        # Preprocess the input
        input_data_scaled = scaler.transform(input_data)
        

        # Make prediction
        prediction = kmeans.predict(input_data_scaled)[0]
        result = "Good Credit" if prediction == 1 else "Bad Credit"

        return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
