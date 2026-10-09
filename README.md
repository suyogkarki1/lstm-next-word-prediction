# NextWord Keyboard (LSTM)

A next-word prediction LSTM (PyTorch) with a phone-style keyboard web UI that suggests the next word as you type.

## Project structure

```
LSTM/
├── app.py                  # Flask server (UI + /predict API)
├── train.py                # Trains the model, saves to artifacts/
├── nextword/
│   └── model.py            # Tokenizer, LSTM model, Predictor
├── data/
│   └── corpus.txt          # Training text
├── artifacts/
│   └── lstm_next_word.pt   # Trained model + vocab
├── templates/
│   └── index.html          # Keyboard page
├── static/
│   ├── css/keyboard.css
│   └── js/keyboard.js
├── notebooks/
│   └── LSTM.ipynb          # Original exploration notebook
├── requirements.txt
└── venv/                   # Virtual environment (not committed)
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name lstm-venv --display-name "Python (LSTM venv)"
```

## Usage

```powershell
python train.py   # retrain after editing data/corpus.txt
python app.py     # open http://127.0.0.1:5000
```

In the UI: tap a suggestion (or press **Tab**) to accept it, and **Enter** to send.
