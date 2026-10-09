from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nextword.model import LSTM, build_vocab, text_to_indices, tokenize

ROOT = Path(__file__).parent
CORPUS = ROOT / 'data' / 'corpus.txt'
CHECKPOINT = ROOT / 'artifacts' / 'lstm_next_word.pt'

EPOCHS = 50
LR = 0.001
BATCH_SIZE = 32


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def main():
    torch.manual_seed(42)
    document = CORPUS.read_text(encoding='utf-8')
    vocab, counts = build_vocab(document)
    print(f'Vocab size: {len(vocab)}')

    # every prefix of every line becomes a training sequence
    training_sequences = []
    for sentence in document.split('\n'):
        indices = text_to_indices(tokenize(sentence), vocab)
        for i in range(1, len(indices)):
            training_sequences.append(indices[:i + 1])

    max_len = max(len(s) for s in training_sequences)
    padded = torch.tensor([[0] * (max_len - len(s)) + s for s in training_sequences], dtype=torch.long)
    X, y = padded[:, :-1], padded[:, -1]
    print(f'Training sequences: {X.shape}')

    loader = DataLoader(CustomDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1:>2}/{EPOCHS}  loss: {total_loss:.4f}')

    CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save({
        'state_dict': model.cpu().state_dict(),
        'vocab': vocab,
        'counts': dict(counts),
        'seq_len': max_len - 1,
    }, CHECKPOINT)
    print(f'Saved model to {CHECKPOINT}')


if __name__ == '__main__':
    main()
