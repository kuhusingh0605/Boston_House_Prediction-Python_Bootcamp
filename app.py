from flask import Flask, render_template, request
import pickle 
import numpy as np

app = Flask(__name__)
# Load the pre-trained model
with open('house_price_prediction.pkl', 'rb') as file:
    model= pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [ 
        float(request.form['CRIM']),  # Crime rate        
        float(request.form['ZN']),    # Residential land zoned for lots over 25,000 sq.ft.
        float(request.form['INDUS']),
        float(request.form['CHAS']),  # Charles River dummy variable (1 if tract bounds river; 0 otherwise)
        float(request.form['NOX']),   # Nitric oxides concentration (parts per 10 million)
        float(request.form['RM']),    # Average number of rooms per dwelling     
        float(request.form['AGE']),   # Proportion of owner-occupied units built prior to 1940
        float(request.form['DIS']),   # Weighted distances to five Boston employment centres
        float(request.form['RAD']),   # Index of accessibility to radial highways
        float(request.form['TAX']),   # Full-value property-tax rate per $10,000
        float(request.form['PTRATIO']),  # Pupil-teacher ratio by town
        float(request.form['B']),      # 1000(Bk - 0.63)^2 where Bk is the proportion
        float(request.form['LSTAT'])  # Percentage of lower status of the population
    ]

    features_array = np.array(features).reshape(1, -1) # <--- This ensures a 2D array
    # Make prediction

    prediction = model.predict(features_array)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted House Price: ${output}')
if __name__ == "__main__":
    app.run(debug=True)
