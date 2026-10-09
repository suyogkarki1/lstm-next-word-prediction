from pathlib import Path

from flask import Flask, jsonify, render_template, request

from nextword.model import Predictor

CHECKPOINT = Path(__file__).parent / 'artifacts' / 'lstm_next_word.pt'

if not CHECKPOINT.exists():
    raise SystemExit('No trained model found. Run `python train.py` first.')

app = Flask(__name__)
predictor = Predictor(CHECKPOINT)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = (request.get_json(silent=True) or {}).get('text', '')
    return jsonify(predictor.suggest(text, k=3))


if __name__ == '__main__':
    app.run(debug=False, port=5000)
