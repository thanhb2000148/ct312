import sys
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Thiết lập mã hóa đầu ra là UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Đọc dữ liệu từ file iris.csv
data = pd.read_csv('iris.csv')

# 2. Tiền xử lý dữ liệu
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Huấn luyện mô hình Decision Tree
model = DecisionTreeClassifier(criterion="gini", random_state=42)
model.fit(X_train, y_train)

# Dự đoán nhãn tập kiểm tra
y_pred = model.predict(X_test)

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy:.3f}")

# 4. Lưu mô hình
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Mô hình đã được lưu vào file decision_tree_model.pkl")
