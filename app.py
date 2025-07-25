from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model and preprocessor
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        try:
            input_data = {
                'gender': request.form['gender'],
                'race_ethnicity': request.form['race_ethnicity'],
                'parental_level_of_education': request.form[
                    'parental_level_of_education'],
                'lunch': request.form['lunch'],
                'test_preparation_course': request.form[
                    'test_preparation_course'],
                'reading_score': float(request.form['reading_score']),
                'writing_score': float(request.form['writing_score'])
            }

            df = pd.DataFrame([input_data])
            transformed = preprocessor.transform(df)
            prediction = model.predict(transformed)[0]
            prediction = round(prediction, 2)

        except Exception as e:
            prediction = f"❌ Error: {e}"

    return render_template("index.html", prediction=prediction)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
