import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from nextword.model import Predictor

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / 'artifacts' / 'lstm_next_word.pt'
METRICS = ROOT / 'reports' / 'metrics.json'

if not CHECKPOINT.exists():
    raise SystemExit('No trained model found. Run `python train.py` first.')

app = Flask(__name__)
predictor = Predictor(CHECKPOINT)
metrics = json.loads(METRICS.read_text()) if METRICS.exists() else None


@app.route('/')
def index():
    return render_template('index.html', metrics=metrics)


@app.route('/predict', methods=['POST'])
def predict():
    text = (request.get_json(silent=True) or {}).get('text', '')
    return jsonify(predictor.suggest(text, k=3))


if __name__ == '__main__':
    app.run(debug=False, port=5000)
