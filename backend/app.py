from flask import Flask, request, jsonify # type: ignore
import pickle
import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Tải mô hình
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['features']])
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    prediction = model.predict(features_scaled)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
