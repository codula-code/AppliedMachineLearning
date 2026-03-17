from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)
model = joblib.load('best_model.joblib')
THRESHOLD = 0.5


@app.route('/score', methods=['POST'])
def score_endpoint():
    data = request.get_json()
    text = data['text']
    prediction, propensity = score(text, model, THRESHOLD)
    return jsonify({
        'prediction': prediction,
        'propensity': propensity
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
