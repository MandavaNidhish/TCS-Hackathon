from flask import Flask, render_template, request
import numpy as np
import joblib
from collections import Counter

app = Flask(__name__)

# Load the saved models and preprocessors
kmeans = joblib.load('kmeans_model.joblib')
gmm = joblib.load('gmm_model.joblib')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca_model.joblib')
cluster_labels_map = joblib.load('cluster_labels_map.joblib')  # Mapping for interpreting cluster as Good/Bad

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            age = float(request.form['age'])
            sex = int(request.form['sex'])
            job = int(request.form['job'])
            housing = int(request.form['housing'])
            saving_accounts = int(request.form['saving_accounts'])
            checking_account = int(request.form['checking_account'])
            purpose = int(request.form['purpose'])
            credit_amount = float(request.form['credit_amount'])
            duration = float(request.form['duration'])

            # Create input array
            input_data = np.array([[age, sex, job, housing, saving_accounts, checking_account,
                                    purpose, credit_amount, duration]])

            # Apply same preprocessing: Scaling → PCA
            scaled_input = scaler.transform(input_data)
            reduced_input = pca.transform(scaled_input)

            # Predict with both models
            pred_kmeans = kmeans.predict(reduced_input)[0]
            pred_gmm = gmm.predict(reduced_input)[0]

            # Ensemble: Majority Voting
            ensemble_prediction = Counter([pred_kmeans, pred_gmm]).most_common(1)[0][0]
            interpreted_label = cluster_labels_map[ensemble_prediction]

            result = "Bad Credit" if interpreted_label == 1 else "Good Credit"
            return render_template('index.html', result=result)

        except Exception as e:
            return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
