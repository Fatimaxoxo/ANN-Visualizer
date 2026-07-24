from flask import Flask, jsonify, request
from flask_cors import CORS
from ann.perceptron_service import run_perceptron_forward
from ann.backprop_service import run_backpropagation

app = Flask(__name__)
CORS(app)

@app.get("/")
def home():
    return jsonify({"message": "ANN Visualizer backend is running"})

@app.post("/api/perceptron/forward")
def perceptron_forward():
    payload = request.get_json()
    inputs = payload.get("inputs", [])
    weights = payload.get("weights", [])
    bias = payload.get("bias", 0.0)
    activation = payload.get("activation", "sigmoid")

    result = run_perceptron_forward(inputs, weights, bias, activation)
    return jsonify(result)

@app.post("/api/perceptron/backprop")
def perceptron_backprop():
    payload = request.get_json()
    inputs = payload.get("inputs", [])
    weights_output = payload.get("weights_output", [])
    weights_hidden = payload.get("weights_hidden", [])
    target = payload.get("target", 0.0)
    learning_rate = payload.get("learning_rate", 0.1)

    result = run_backpropagation(inputs, weights_output, weights_hidden, target, "sigmoid", "sigmoid", learning_rate)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
