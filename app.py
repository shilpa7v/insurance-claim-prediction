import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

# Function to make predictions
def ValuePredictor(new_l): 
    to_predict = np.array(new_l).reshape(1, 7)
    loaded_model = pickle.load(open("model.pkl", "rb"))  # Load pre-trained model
    result = loaded_model.predict(to_predict)
    return result[0]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home_():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    prediction = ''
    if request.method == 'POST':
        # Retrieve form data
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        new_l = []
        
        # Convert categorical data to numeric values
        for x in to_predict_list:
            if x == 'male':
                new_l.append(1)
            elif x == 'female':
                new_l.append(0)
            elif x == 'yes':
                new_l.append(1)
            elif x == 'no':
                new_l.append(0)
            else:
                new_l.append(x)
        
        # Convert to float and make prediction
        new_l = list(map(float, new_l))
        result = ValuePredictor(new_l)
        
        if int(result) == 1:
            prediction = 'Yes, the person can claim insurance.'
        else:
            prediction = 'No, the person cannot claim insurance.'

    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
