from pathlib import Path

from flask import Flask, jsonify, request

from nextword.model import Predictor

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / 'artifacts' / 'lstm_next_word.pt'

if not CHECKPOINT.exists():
    raise SystemExit('No trained model found. Run `python train.py` first.')

# the same frontend/ folder is also used by the Streamlit app (streamlit_app.py)
app = Flask(__name__, static_folder='frontend', static_url_path='')
predictor = Predictor(CHECKPOINT)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = (request.get_json(silent=True) or {}).get('text', '')
    return jsonify(predictor.suggest(text, k=3))


@app.route('/continue', methods=['POST'])
def continue_text():
    text = (request.get_json(silent=True) or {}).get('text', '')
    words, finished = predictor.generate(text)
    return jsonify({'words': words, 'finished': finished})


if __name__ == '__main__':
    app.run(debug=False, port=5000)
