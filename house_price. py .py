# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from flask import Flask, request, jsonify
import threading
import requests

# Step 1: Create dataset
data = pd.DataFrame({
    "Rooms": [1,2,3,2,4,3,5],
    "Area": [400,600,800,700,1200,1000,1500],
    "Age": [10,15,20,5,30,10,25],
    "Price": [200000,250000,300000,270000,500000,400000,600000]
})

print("Dataset:")
print(data)

# Step 2: Split data
X = data[["Rooms","Area","Age"]]
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining rows:", X_train.shape[0], "Testing rows:", X_test.shape[0])

# Step 3: Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nPredicted Prices for Test Data:", y_pred)
print("Mean Squared Error:", mse)

# Step 4: Save model
joblib.dump(model, "house_price_model.pkl")
print("\nModel saved successfully!")

# Step 5: Flask API
app = Flask(__name__)
model = joblib.load("house_price_model.pkl")

@app.route('/')
def home():
    return "House Price Prediction API Running in Spyder Only"

@app.route('/predict', methods=['POST'])
def predict():
    data_input = request.get_json(force=True)
    input_array = np.array([data_input['Rooms'], data_input['Area'], data_input['Age']]).reshape(1,-1)
    prediction = model.predict(input_array)
    return jsonify({"Predicted_Price": float(prediction[0])})

# Step 6: Run Flask in a thread (so we can test in Spyder console)
def run_flask():
    app.run(debug=False, use_reloader=False)

thread = threading.Thread(target=run_flask)
thread.start()

# Step 7: Test API from Spyder itself
test_input = {"Rooms":3, "Area":900, "Age":10}
response = requests.post("http://127.0.0.1:5000/predict", json=test_input)
print("\nTest Input:", test_input)
print("Predicted Price from API:", response.json())
