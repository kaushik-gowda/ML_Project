from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load preprocessor and model
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))
model = pickle.load(open("artifacts/model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        input_data = {
            'gender': request.form['gender'],
            'race_ethnicity': request.form['race_ethnicity'],
            'parental_level_of_education': request.form[
                'parental_level_of_education'],
            'lunch': request.form['lunch'],
            'test_preparation_course': request.form['test_preparation_course'],
            'reading_score': float(request.form['reading_score']),
            'writing_score': float(request.form['writing_score'])
        }

        df = pd.DataFrame([input_data])
        transformed_input = preprocessor.transform(df)
        prediction = model.predict(transformed_input)

        return render_template('result.html', prediction=round(prediction[0],
                                                               2))

    except Exception as e:
        return f"❌ Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)
