"""Stage 1: pretrain the LSTM as a general English language model on WikiText-2.

The text is one long token stream cut into batches of BPTT-length chunks, and the model
predicts the next word at every position (truncated backpropagation through time).
Output: artifacts/pretrained_lstm.pt, used by train.py for fine-tuning.
"""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from nextword.model import LSTM, UNK_IDX, text_to_indices
from nextword.wikitext import build_general_vocab, load_sentences

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / 'artifacts' / 'pretrained_lstm.pt'
REPORTS = ROOT / 'reports'

SEED = 42
VOCAB_SIZE = 12000
MODEL_CONFIG = dict(embed_dim=256, hidden_dim=256, num_layers=1, dropout=0.3, tie_weights=True)
BATCH_SIZE = 64
BPTT = 35
LR = 0.002
EPOCHS = 4


def to_stream(sentences, vocab):
    ids = [i for s in sentences for i in text_to_indices(s, vocab)]
    return torch.tensor(ids, dtype=torch.long)


def batchify(stream, batch_size):
    n = stream.size(0) // batch_size
    return stream[:n * batch_size].view(batch_size, n)          # (batch, time)


def run_epoch(model, data, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    state, total, count = None, 0.0, 0
    with torch.set_grad_enabled(training):
        for i in range(0, data.size(1) - 1, BPTT):
            seq = min(BPTT, data.size(1) - 1 - i)
            x, y = data[:, i:i + seq], data[:, i + 1:i + 1 + seq]
            if state is not None:
                state = tuple(s.detach() for s in state)       # truncated BPTT
            logits, state = model.forward_all(x, state)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            n = (y != UNK_IDX).sum().item()
            total += loss.item() * n
            count += n
    return total / count


def main():
    torch.manual_seed(SEED)
    train_s, val_s = load_sentences('train'), load_sentences('validation')
    vocab = build_general_vocab(train_s, VOCAB_SIZE)
    train, val = batchify(to_stream(train_s, vocab), BATCH_SIZE), batchify(to_stream(val_s, vocab), BATCH_SIZE)
    print(f'WikiText-2: {train.numel():,} train tokens, {val.numel():,} val tokens, vocab {len(vocab):,}')

    model = LSTM(len(vocab), **MODEL_CONFIG)
    criterion = nn.CrossEntropyLoss(ignore_index=UNK_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)

    history = []
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = run_epoch(model, train, criterion, optimizer)
        val_loss = run_epoch(model, val, criterion)
        scheduler.step()
        history.append({'epoch': epoch, 'train_ppl': round(math.exp(train_loss), 2),
                        'val_ppl': round(math.exp(val_loss), 2)})
        print(f'Epoch {epoch}  train ppl {math.exp(train_loss):7.1f}  val ppl {math.exp(val_loss):7.1f}'
              f'  ({time.time() - start:.0f}s)', flush=True)

    CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'model_config': MODEL_CONFIG}, CHECKPOINT)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / 'pretrain_metrics.json').write_text(json.dumps(
        {'dataset': 'WikiText-2 (raw)', 'train_tokens': train.numel(), 'val_tokens': val.numel(),
         'vocab_size': len(vocab), 'model': MODEL_CONFIG, 'batch_size': BATCH_SIZE, 'bptt': BPTT,
         'lr': LR, 'history': history}, indent=2))
    print(f'Saved pretrained model to {CHECKPOINT}')


if __name__ == '__main__':
    main()
