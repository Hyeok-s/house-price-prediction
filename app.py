from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# 모델 로드
best_forest_model = joblib.load('best_forest_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 사용자로부터 특성 입력 받기
        features = [
            float(request.form['feature{}'.format(i)]) for i in range(1, 17)
        ]

        # 모델 예측
        predicted_price = best_forest_model.predict([features])[0]

        return render_template('result.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
