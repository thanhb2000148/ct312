import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Đọc mô hình đã lưu
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Tạo giao diện
def classify():
    try:
        # Lấy dữ liệu từ các ô nhập
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())
        
        # Chuẩn bị dữ liệu để dự đoán
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        
        # Dự đoán
        prediction = model.predict(data)
        messagebox.showinfo("Kết quả phân lớp", f"Loài dự đoán: {prediction[0]}")
    except ValueError:
        messagebox.showerror("Lỗi dữ liệu", "Vui lòng nhập dữ liệu hợp lệ.")

app = tk.Tk()
app.title("Ứng dụng phân lớp Iris")

tk.Label(app, text="Chiều dài đài hoa (sepal length):").grid(row=0, column=0)
tk.Label(app, text="Chiều rộng đài hoa (sepal width):").grid(row=1, column=0)
tk.Label(app, text="Chiều dài cánh hoa (petal length):").grid(row=2, column=0)
tk.Label(app, text="Chiều rộng cánh hoa (petal width):").grid(row=3, column=0)

entry_sepal_length = tk.Entry(app)
entry_sepal_width = tk.Entry(app)
entry_petal_length = tk.Entry(app)
entry_petal_width = tk.Entry(app)

entry_sepal_length.grid(row=0, column=1)
entry_sepal_width.grid(row=1, column=1)
entry_petal_length.grid(row=2, column=1)
entry_petal_width.grid(row=3, column=1)

tk.Button(app, text="Dự đoán", command=classify).grid(row=4, column=0, columnspan=2)

app.mainloop()
