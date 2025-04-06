import joblib
from flask import Flask, request, jsonify, render_template

# Load the saved pipeline (includes vectorizer + SVM)
svm_pipeline = joblib.load("svm_pipeline.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None  # Default sentiment result

    if request.method == "POST":
        review_text = request.form.get("review_text")  # Get input from form

        if review_text:  # Ensure text is not empty
            sentiment = svm_pipeline.predict([review_text])[0]  # Predict sentiment

    return render_template("index.html", sentiment=sentiment)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Expecting JSON request
    review_text = data.get("text")

    if not review_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        sentiment = svm_pipeline.predict([review_text])[0]  # Directly use model prediction
        return jsonify({"sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
