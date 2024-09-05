from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your CSV dataset (replace 'your_data.csv' with the actual file name)
data = pd.read_csv('combined_output.csv')

# Load your trained machine learning model (replace 'model.pkl' with the actual model file name)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Home route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        input_id = int(request.form['id'])  # Convert input ID to integer
        difficulty = request.form['difficulty']

        # Find the row corresponding to the ID
        row = data[data['ID'] == input_id]

        # Check if the ID exists in the dataset
        if row.empty:
            return render_template('index.html', message="ID not found in dataset.")
        
        # Check if the difficulty is valid
        if difficulty.lower() not in ['easy', 'medium', 'hard']:
            return render_template('index.html', message="Please enter a valid difficulty: Easy, Medium, or Hard.")
        
        # Convert difficulty to the numerical form expected by the model (assume encoded as 0, 1, 2)
        difficulty_encoded = {'easy': 0, 'medium': 1, 'hard': 2}[difficulty.lower()]

        # Prepare the feature vector for the prediction (you can add more features if needed)
        X_input = np.array([[difficulty_encoded]])  # Modify as per the features used in your model
        
        # Make prediction (predicting Frequency and Acceptance)
        prediction = model.predict(X_input)

        # Extract predictions
        predicted_frequency = prediction[0][0]
        predicted_acceptance = prediction[0][1]

        predicted_acceptance_percentage = predicted_acceptance * 100

        # Render the result template with the predicted values
        return render_template('result.html', frequency=predicted_frequency, acceptance=predicted_acceptance_percentage)

    except Exception as e:
        return
    

if __name__ == '__main__':
    app.run(debug=True,port=5000)
