from flask import Flask,render_template, request
import pickle
import numpy as np

model = pickle.load(open('iris_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
# Updated code for /predict route
@app.route('/predict', methods=['POST'])
def predict(): # Renamed the function to 'predict'
    data1 = request.form.get('a')
    data2 = request.form.get('b')
    data3 = request.form.get('c')
    data4 = request.form.get('d')
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

@app.route('/test')
def test_iris():
    return 'This is not home page '

if __name__ == "__main__":
    app.run(debug=True)