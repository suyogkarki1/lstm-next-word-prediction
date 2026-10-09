# NextWord Keyboard (LSTM)

A personal next-word prediction keyboard. A PyTorch LSTM is **pretrained on general English** (WikiText-2, 1.9M words), then **fine-tuned on my own writing** (CV, projects and an "about me"). It's served through Flask to an iPhone-style keyboard with a suggestion bar, grey inline completion, and a chat where the LSTM finishes your sentences.

![Model comparison](reports/model_comparison.png)

## Results

All models are evaluated with **5-fold cross-validation over my personal sentences**. In every fold, each model is scored on exactly the same held-out words:

| Model | Top-3 accuracy | Top-1 accuracy | Perplexity ↓ | Knows the word |
|---|---|---|---|---|
| **Pretrained + fine-tuned** | **29.4% ± 1.7** | **18.6% ± 2.4** | **78.9 ± 17.4** | **90.5%** |
| Bigram baseline | 26.4% ± 2.8 | 16.2% ± 1.9 | 84.6 ± 15.3 | 72.7% |
| LSTM, personal data only | 25.0% ± 1.9 | 16.1% ± 1.7 | 87.2 ± 13.8 | 72.7% |
| Pretrained, no fine-tuning | 15.5% ± 1.1 | 7.5% ± 0.4 | 485.3 ± 45.7 | 79.3% |

- **Top-k accuracy** is measured over *every* test word. A word a model can't produce counts as a miss, so all models share the same denominator.
- **Perplexity** is measured over the words every model knows, so the numbers are directly comparable.
- **Knows the word** is the share of test words in the model's vocabulary.

**What this shows**
- **Transfer learning works.** Pretraining plus fine-tuning is best on every metric. On top-3 it beats both baselines in 4 of 5 folds and ties the bigram in the other.
- **The gain comes from vocabulary and language knowledge.** Trained on my ~100 sentences alone, an LSTM can't beat a bigram: about 27% of test words never appear in training. The pretrained model already knows 90% of them, and fine-tuning teaches it my phrasing.
- **Either stage alone is not enough.** The pretrained model by itself writes like Wikipedia: "i love to" → *be, the, have*. After fine-tuning: "I love to" → *visit, play, hike*.

![Training curves](reports/loss_curve.png)

The training curves show overfitting that early stopping catches: validation loss bottoms out while training loss keeps falling. The two panels' loss values are not directly comparable. The personal-only model's loss skips the ~27% of words outside its small vocabulary, while the fine-tuned model is scored on nearly all of them.

### Sample suggestions (shipped model)

| You type | Suggestions |
|---|---|
| `I love to` | visit · play · hike |
| `I love to play` | football · table · a |
| `I love to visit the` | mountains · new · project |
| `Hiking clears my` | mind · friends · and |
| `I built a diabetes risk` | predictor · of · on |

## How it works

**Stage 1: pretraining** (`pretrain.py`)
- **Data**: WikiText-2 (raw), 1.87M training tokens, downloaded automatically. Vocabulary is the 12,000 most frequent words.
- **Model**: Embedding(256) → LSTM(256) → Linear, with **tied input/output embeddings** (Press & Wolf, 2017) and dropout 0.3.
- **Training**: language-model training on the token stream. Batch 64, truncated backprop through time over 35 steps, Adam with a step-decayed learning rate, 4 epochs. WikiText validation perplexity went 425 → 351 → 321 → **305**.

**Stage 2: fine-tuning** (`train.py`)
- **Vocabulary**: the pretrained vocabulary is extended with my words. Known words keep their trained vectors, and new ones start random.
- **Examples**: every prefix of every sentence becomes one (context → next word) example. Contexts are left-padded to 30 tokens, and the LSTM **skips padding entirely** using packed sequences.
- **Training**: learning rate 1e-3 (lower than pretraining, so knowledge is adapted rather than overwritten), dropout 0.5, early stopping on a validation slice.
- **Shipped model**: fine-tuned on all sentences for the median best epoch found in cross-validation (18).

**Data handling**
- Contact details and URLs are removed from the personal corpus.
- In each fold, the vocabulary comes from training sentences only.
- The bigram baseline is interpolated with a unigram distribution (λ = 0.7).

## Project structure

```
LSTM/
├── app.py                  # Flask server: page, /predict, /continue
├── pretrain.py             # stage 1: WikiText-2 language model
├── train.py                # stage 2: fine-tuning + 4-way cross-validation
├── nextword/
│   ├── model.py            # tokenizer, LSTM, vocab expansion, Predictor
│   └── wikitext.py         # WikiText-2 download + preprocessing
├── data/corpus.txt         # personal training text
├── artifacts/
│   ├── pretrained_lstm.pt  # stage 1 model
│   └── lstm_next_word.pt   # shipped model
├── reports/                # metrics.json, pretrain_metrics.json, charts
├── templates/index.html    # keyboard page
├── static/                 # css + js
└── notebooks/LSTM.ipynb    # original exploration notebook
```

## Setup and usage

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

python pretrain.py   # optional, ~35 min on CPU: the pretrained model is already in artifacts/
python train.py      # ~12 min: cross-validation + final fine-tuning
python app.py        # open http://127.0.0.1:5000
```

In the UI:
- Tap a suggestion, or press **Tab**, to accept it. Grey inline text shows the top prediction.
- Press **Enter** to send. The LSTM replies by continuing your sentence.

## Limitations and next steps

- **More personal text** is still the biggest lever, since only ~100 sentences are used for fine-tuning.
- **Greedy generation** in chat replies tends to drift and repeat. Beam search or nucleus sampling would help.
- **Subword tokenization** (BPE) would let the model handle words it has never seen.
- **A small Transformer** pretrained the same way would be a natural comparison.
