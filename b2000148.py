from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Đọc mô hình đã lưu
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ yêu cầu
        data = request.json
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
        
        # Chuẩn bị dữ liệu để dự đoán
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)
        
        # Dự đoán
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction[0]})
    except ValueError:
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
