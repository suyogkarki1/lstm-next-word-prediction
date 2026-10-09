import re
from collections import Counter

import nltk
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# characters stripped before tokenizing (bullets / separators from the resume layout)
NOISE = [',', '·', '•']


def clean(text):
    text = text.lower()
    for ch in NOISE:
        text = text.replace(ch, ' ')
    return text


def tokenize(text):
    return word_tokenize(clean(text))


def build_vocab(document):
    tokens = tokenize(document)
    vocab = {'<unk>': 0}
    for token in Counter(tokens).keys():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab, Counter(tokens)


def text_to_indices(tokens, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokens]


class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, X):
        embedded = self.embedding(X)
        intermediate_hidden_state, (final_hidden_state, final_cell_state) = self.lstm(embedded)
        return self.fc(final_hidden_state.squeeze(0))


def is_word(token):
    # suggestions should be words/numbers, not lone punctuation
    return bool(re.search(r'[a-z0-9]', token))


class Predictor:
    def __init__(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.vocab = ckpt['vocab']
        self.idx_to_word = {i: w for w, i in self.vocab.items()}
        self.counts = Counter(ckpt['counts'])
        self.seq_len = ckpt['seq_len']
        self.model = LSTM(len(self.vocab))
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    @torch.no_grad()
    def _next_word_probs(self, context_tokens):
        indices = text_to_indices(context_tokens, self.vocab)[-self.seq_len:]
        padded = [0] * (self.seq_len - len(indices)) + indices
        logits = self.model(torch.tensor([padded], dtype=torch.long))
        return torch.softmax(logits, dim=1)[0]

    def suggest(self, text, k=3):
        """Top-k suggestions. If the text ends mid-word, complete that word instead."""
        partial = ''
        context = text
        if text and not text[-1].isspace():
            match = re.search(r'(\S+)$', text)
            partial = clean(match.group(1)).strip()
            context = text[:match.start()]

        probs = self._next_word_probs(tokenize(context))
        ranked = torch.argsort(probs, descending=True).tolist()

        results = []
        for idx in ranked:
            word = self.idx_to_word[idx]
            if idx == 0 or not is_word(word) or not word.startswith(partial):
                continue
            results.append({'word': word, 'score': round(probs[idx].item(), 4)})
            if len(results) == k:
                break
        return {'partial': partial, 'suggestions': results}
