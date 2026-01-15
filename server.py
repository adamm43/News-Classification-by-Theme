# backend/server.py
import os
import pickle
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from utils import clean_text


app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load artifacts safely
MODEL_PATH = os.path.join("artifacts", "model.pkl")
VECT_PATH = os.path.join("artifacts", "vectorizer.pkl")

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECT_PATH, "rb"))
    print("✅ Model & vectorizer loaded.")
except Exception as e:
    model = None
    vectorizer = None
    print("⚠ Could not load artifacts:", e)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

# Serve frontend index (if frontend folder exists next to backend/)
@app.route("/app", methods=["GET"])
def serve_front():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
    frontend_path = os.path.abspath(frontend_path)
    index_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_file):
        return send_from_directory(frontend_path, "index.html")
    return jsonify({"error": "frontend not found"}), 404

# Serve static assets if frontend folder exists
@app.route("/app/<path:filename>")
def serve_static(filename):
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
    frontend_path = os.path.abspath(frontend_path)
    file_path = os.path.join(frontend_path, filename)
    if os.path.exists(file_path):
        return send_from_directory(frontend_path, filename)
    abort(404)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running. Visit /app for the frontend (if present)"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "model_not_loaded"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid_json"}), 400

    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "empty_text"}), 400

    try:
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(max(model.predict_proba(X)[0]))
        return jsonify({
            "prediction": pred,
            "confidence": round(prob * 100, 2) if prob is not None else None
        })
    except Exception as e:
        return jsonify({"error": "inference_error", "detail": str(e)}), 500

if __name__ == "__main__":
    # use_reloader=False to avoid infinite reloads on OneDrive / Windows
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
