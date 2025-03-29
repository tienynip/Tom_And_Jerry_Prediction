import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load mô hình
model = joblib.load("model.pkl")
categories = ['Jerry', 'Tom']  # 0 = Jerry, 1 = Tom
img_size = 50

def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size))
    img = img.flatten()
    img = np.array(img).reshape(1, -1)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Xử lý và dự đoán ảnh
        img = process_image(file_path)
        if img is None:
            return render_template('index.html', message='Invalid image')
        
        label = model.predict(img)[0]
        predicted_label = categories[label]
        
        return render_template('index.html', image_url=file_path, result=predicted_label, css_url='static/style.css')
    
    return render_template('index.html', css_url='static/style.css')

if __name__ == '__main__':
    app.run(debug=True)
