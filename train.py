"""Train the next-word LSTM.

1. Evaluate: 5-fold cross-validation over sentences. In each fold the model trains on
   4/5 of the sentences (with a validation slice for early stopping) and is scored on the
   held-out 1/5, next to a bigram baseline trained on the same data.
2. Ship: train one final model on the full corpus for the keyboard app. This is a personal
   keyboard, so the deployed model is meant to learn all of your writing.
"""
import copy
import json
import math
import random
import statistics
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nextword.model import (LSTM, PAD_IDX, UNK_IDX, build_vocab, make_examples,
                            split_sentences, text_to_indices)

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
CORPUS = ROOT / 'data' / 'corpus.txt'
CHECKPOINT = ROOT / 'artifacts' / 'lstm_next_word.pt'
REPORTS = ROOT / 'reports'

SEED = 42
SEQ_LEN = 30
# picked with a small validation sweep: on ~100 sentences, a small single-layer model
# with strong dropout + weight decay generalizes best (bigger models overfit within ~6 epochs)
MODEL_CONFIG = dict(embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.5)
LR = 0.002
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 150
PATIENCE = 15
K_FOLDS = 5
VAL_FRACTION = 0.1
FINAL_EPOCHS = 60
TOP_K = (1, 3)


def scores_from_logits(scores, y, log_probs):
    """Perplexity + top-k accuracy, ignoring targets that are out of vocabulary."""
    keep = y != UNK_IDX
    scores, y, log_probs = scores[keep], y[keep], log_probs[keep]
    nll = -log_probs.gather(1, y[:, None]).mean().item()
    scores = scores.clone()
    scores[:, [PAD_IDX, UNK_IDX]] = -float('inf')          # never suggest these
    ranked = scores.topk(max(TOP_K), dim=1).indices
    result = {'perplexity': math.exp(nll)}
    for k in TOP_K:
        result[f'top{k}_acc'] = (ranked[:, :k] == y[:, None]).any(1).float().mean().item()
    return result


@torch.no_grad()
def evaluate_lstm(model, X, y):
    model.eval()
    logits = torch.cat([model(xb) for xb in X.split(512)])
    return scores_from_logits(logits, y, torch.log_softmax(logits, 1))


def bigram_baseline(train, test, vocab, lam=0.7):
    """Interpolated bigram: P(w | prev) = lam * P_bigram + (1 - lam) * P_unigram (add-one)."""
    V, BOS = len(vocab), len(vocab)                        # extra row for "start of sentence"
    pair, uni = torch.zeros(V + 1, V), torch.ones(V)
    for tokens in train:
        idx = text_to_indices(tokens, vocab)
        for i, t in enumerate(idx):
            pair[idx[i - 1] if i else BOS, t] += 1
            uni[t] += 1
    p_uni = uni / uni.sum()
    row = pair.sum(1, keepdim=True)
    probs = lam * torch.where(row > 0, pair / row.clamp(min=1), p_uni) + (1 - lam) * p_uni

    prev, y = [], []
    for tokens in test:
        idx = text_to_indices(tokens, vocab)
        for i, t in enumerate(idx):
            prev.append(idx[i - 1] if i else BOS)
            y.append(t)
    p = probs[torch.tensor(prev)]
    return scores_from_logits(p, torch.tensor(y), p.log())


def fit(X_train, y_train, vocab_size, X_val=None, y_val=None, epochs=MAX_EPOCHS, verbose=False):
    """Train an LSTM. With a validation set: early stopping, returns the best epoch's weights."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(vocab_size, **MODEL_CONFIG).to(device)
    criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss(ignore_index=UNK_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    history = {'train_loss': [], 'val_loss': []}
    best = {'loss': float('inf'), 'epoch': 0, 'state': None}

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch_y)
        history['train_loss'].append(total / len(y_train))

        if X_val is None:
            scheduler.step(history['train_loss'][-1])
            if verbose and epoch % 10 == 0:
                print(f'  epoch {epoch:>3}  train {history["train_loss"][-1]:.4f}')
            continue

        model.eval()
        with torch.no_grad():
            val_loss = val_criterion(model(X_val.to(device)), y_val.to(device)).item()
        scheduler.step(val_loss)
        history['val_loss'].append(val_loss)
        if val_loss < best['loss']:
            best = {'loss': val_loss, 'epoch': epoch, 'state': copy.deepcopy(model.state_dict())}
        if epoch - best['epoch'] >= PATIENCE:
            break

    if best['state'] is not None:
        model.load_state_dict(best['state'])
    return model.cpu(), best['epoch'], history


def cross_validate(sentences):
    order = list(range(len(sentences)))
    random.Random(SEED).shuffle(order)
    folds = [order[i::K_FOLDS] for i in range(K_FOLDS)]
    results, histories = [], []

    for k, test_idx in enumerate(folds, 1):
        rest = [i for i in order if i not in set(test_idx)]
        n_val = max(1, int(len(rest) * VAL_FRACTION))
        val_s = [sentences[i] for i in rest[:n_val]]
        train_s = [sentences[i] for i in rest[n_val:]]
        test_s = [sentences[i] for i in test_idx]

        vocab = build_vocab(train_s)                      # vocab from training data only
        X_tr, y_tr = make_examples(train_s, vocab, SEQ_LEN)
        X_va, y_va = make_examples(val_s, vocab, SEQ_LEN)
        X_te, y_te = make_examples(test_s, vocab, SEQ_LEN)

        torch.manual_seed(SEED + k)
        model, best_epoch, history = fit(X_tr, y_tr, len(vocab), X_va, y_va)
        lstm = evaluate_lstm(model, X_te, y_te)
        bigram = bigram_baseline(train_s, test_s, vocab)
        oov = (y_te == UNK_IDX).float().mean().item()
        results.append({'fold': k, 'best_epoch': best_epoch, 'test_words': len(y_te),
                        'oov_rate': oov, 'lstm': lstm, 'bigram': bigram})
        histories.append(history)
        print(f'Fold {k}: best epoch {best_epoch:>3} | '
              f'LSTM ppl {lstm["perplexity"]:6.1f} top3 {lstm["top3_acc"]:.1%} | '
              f'bigram ppl {bigram["perplexity"]:6.1f} top3 {bigram["top3_acc"]:.1%} | OOV {oov:.1%}')
    return results, histories


def summarize(results, model_name):
    out = {}
    for metric in ('perplexity', 'top1_acc', 'top3_acc'):
        values = [r[model_name][metric] for r in results]
        out[metric] = {'mean': round(statistics.mean(values), 4), 'std': round(statistics.stdev(values), 4)}
    return out


def plot_curves(histories, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=130)
    for i, h in enumerate(histories):
        epochs = range(1, len(h['train_loss']) + 1)
        ax.plot(epochs, h['train_loss'], color='#7c5cff', lw=1.6, alpha=.75, label='train' if i == 0 else None)
        ax.plot(epochs, h['val_loss'], color='#16a3c7', lw=1.6, alpha=.75, label='validation' if i == 0 else None)
    ax.set_xlabel('epoch')
    ax.set_ylabel('cross-entropy loss')
    ax.set_title(f'Training curves across {len(histories)} cross-validation folds')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    sentences = split_sentences(CORPUS.read_text(encoding='utf-8'))
    print(f'{len(sentences)} sentences, {sum(map(len, sentences))} tokens\n')

    print(f'== {K_FOLDS}-fold cross-validation ==')
    results, histories = cross_validate(sentences)
    lstm, bigram = summarize(results, 'lstm'), summarize(results, 'bigram')
    print(f'\n{"":8}{"perplexity":>18}{"top-1":>16}{"top-3":>16}')
    for name, s in (('LSTM', lstm), ('Bigram', bigram)):
        print(f'{name:8}{s["perplexity"]["mean"]:>10.1f} ± {s["perplexity"]["std"]:<5.1f}'
              f'{s["top1_acc"]["mean"]:>9.1%} ± {s["top1_acc"]["std"]:<5.1%}'
              f'{s["top3_acc"]["mean"]:>9.1%} ± {s["top3_acc"]["std"]:<5.1%}')

    print(f'\n== Final model: all {len(sentences)} sentences, {FINAL_EPOCHS} epochs ==')
    vocab = build_vocab(sentences)
    X, y = make_examples(sentences, vocab, SEQ_LEN)
    torch.manual_seed(SEED)
    model, _, _ = fit(X, y, len(vocab), epochs=FINAL_EPOCHS, verbose=True)

    CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'vocab': vocab,
                'seq_len': SEQ_LEN, 'model_config': MODEL_CONFIG}, CHECKPOINT)

    REPORTS.mkdir(exist_ok=True)
    metrics = {
        'data': {'sentences': len(sentences), 'tokens': sum(map(len, sentences)), 'final_vocab_size': len(vocab)},
        'model': {**MODEL_CONFIG, 'seq_len': SEQ_LEN, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
                  'batch_size': BATCH_SIZE, 'patience': PATIENCE, 'final_epochs': FINAL_EPOCHS},
        'cross_validation': {'folds': K_FOLDS, 'lstm': lstm, 'bigram_baseline': bigram,
                             'per_fold': [{**r, 'lstm': {m: round(v, 4) for m, v in r['lstm'].items()},
                                           'bigram': {m: round(v, 4) for m, v in r['bigram'].items()},
                                           'oov_rate': round(r['oov_rate'], 4)} for r in results]},
    }
    (REPORTS / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    plot_curves(histories, REPORTS / 'loss_curve.png')
    print(f'\nSaved model to {CHECKPOINT}')
    print(f'Saved metrics and loss curves to {REPORTS}')


if __name__ == '__main__':
    main()
