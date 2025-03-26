import joblib
from flask import Flask, request, jsonify

# Load the saved pipeline (includes vectorizer + SVM)
svm_pipeline = joblib.load("svm_pipeline.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Sentiment Analysis API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    #print("Received request!")  # Debugging
    data = request.json  # Expecting a JSON request with {'text': 'some review text'}
    #print("Data received:", data)  # Debugging
    if "text" not in data:
        return jsonify({"error": "Missing 'text' key in request"}), 400
    
    # Predict sentiment using the pipeline
    prediction = svm_pipeline.predict([data["text"]])[0]

    # Convert numerical prediction to a label
    # sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    # sentiment = sentiment_map.get(prediction, "Unknown")

    return jsonify({"sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)
