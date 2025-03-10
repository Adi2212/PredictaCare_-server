from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load trained model
model = pickle.load(open('diabetes_model.sav', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Choose correct data source (GET -> query params, POST -> form-data)
        data_source = request.args if request.method == 'GET' else request.form

        # Function to safely convert values to float, defaulting to 0 if missing/invalid
        def safe_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0  # Default to 0 if conversion fails

        # Convert inputs safely
        pregnancies = safe_float(data_source.get('pregnancies', 0))
        glucose = safe_float(data_source.get('glucose', 0))
        BP = safe_float(data_source.get('BP', 0))
        skinthickness = safe_float(data_source.get('skinthickness', 0))
        insulin = safe_float(data_source.get('insulin', 0))
        BMI = safe_float(data_source.get('BMI', 0))
        diabetespedigreefunction = safe_float(data_source.get('diabetespedigreefunction', 0))
        age = safe_float(data_source.get('age', 0))

        # Create NumPy array and reshape it
        input_query = np.array([[pregnancies, glucose, BP, skinthickness, insulin, BMI, diabetespedigreefunction, age]])
        #inputaa= {'pregnancies':pregnancies,'glucose':glucose,'BP':BP,'skinthickness':skinthickness,'insulin':insulin,'BMI':BMI,'diabetespedigreefunction':diabetespedigreefunction,'age':age}
        #return jsonify(inputaa)
        
        # Make prediction
        result = model.predict(input_query)[0]

        return jsonify({'diabetes_prediction': str(result)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


