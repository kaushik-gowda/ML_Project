from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model and preprocessor
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        # Get form data
        gender = request.form["gender"]
        race_ethnicity = request.form["race_ethnicity"]
        parental_level_of_education = request.form["parental_level_of_education"]
        lunch = request.form["lunch"]
        test_preparation_course = request.form["test_preparation_course"]
        reading_score = request.form["reading_score"]
        writing_score = request.form["writing_score"]

        try:
            # Create dataframe for prediction
            input_data = pd.DataFrame([{
                "gender": gender,
                "race_ethnicity": race_ethnicity,
                "parental_level_of_education": parental_level_of_education,
                "lunch": lunch,
                "test_preparation_course": test_preparation_course,
                "reading_score": float(reading_score),
                "writing_score": float(writing_score),
            }])

            # Transform input and predict
            transformed_input = preprocessor.transform(input_data)
            prediction = model.predict(transformed_input)
            result = round(prediction[0], 2)

        except Exception as e:
            result = f"❌ Error: {str(e)}"

    return render_template("index.html", result=result)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
