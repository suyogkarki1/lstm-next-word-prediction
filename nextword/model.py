import re
from collections import Counter

import nltk
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

PAD, UNK = '<pad>', '<unk>'
PAD_IDX, UNK_IDX = 0, 1

# characters stripped before tokenizing (bullets / separators from the resume layout)
NOISE = [',', '·', '•']


def clean(text):
    text = text.lower()
    for ch in NOISE:
        text = text.replace(ch, ' ')
    return text


def tokenize(text):
    return word_tokenize(clean(text))


def split_sentences(document):
    """Each sentence of each non-empty line, tokenized. Sentences are the unit for train/val/test splits."""
    sentences = []
    for line in document.splitlines():
        for sent in sent_tokenize(line.strip()):
            tokens = tokenize(sent)
            if len(tokens) >= 2:
                sentences.append(tokens)
    return sentences


def build_vocab(sentences):
    vocab = {PAD: PAD_IDX, UNK: UNK_IDX}
    for token in Counter(t for s in sentences for t in s):
        vocab[token] = len(vocab)
    return vocab


def text_to_indices(tokens, vocab):
    return [vocab.get(token, UNK_IDX) for token in tokens]


def make_examples(sentences, vocab, seq_len):
    """Every prefix of every sentence -> (left-padded context of seq_len, next word).
    i = 0 is the empty context, so the model also learns how sentences start."""
    X, y = [], []
    for tokens in sentences:
        indices = text_to_indices(tokens, vocab)
        for i in range(len(indices)):
            context = indices[max(0, i - seq_len):i]
            X.append([PAD_IDX] * (seq_len - len(context)) + context)
            y.append(indices[i])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, X):
        embedded = self.dropout(self.embedding(X))
        intermediate_hidden_state, (final_hidden_state, final_cell_state) = self.lstm(embedded)
        # hidden state of the top LSTM layer at the last time step
        return self.fc(self.dropout(final_hidden_state[-1]))


def is_word(token):
    # suggestions should be words/numbers, not lone punctuation
    return bool(re.search(r'[a-z0-9]', token))


class Predictor:
    def __init__(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.vocab = ckpt['vocab']
        self.idx_to_word = {i: w for w, i in self.vocab.items()}
        self.seq_len = ckpt['seq_len']
        self.model = LSTM(len(self.vocab), **ckpt['model_config'])
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    @torch.no_grad()
    def _next_word_probs(self, context_tokens):
        indices = text_to_indices(context_tokens, self.vocab)[-self.seq_len:]
        padded = [PAD_IDX] * (self.seq_len - len(indices)) + indices
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

        # only the current sentence matters, like in training
        sentences = sent_tokenize(context.strip()) if context.strip() else []
        if sentences and re.search(r'[.!?]\s*$', context):
            sentences = []                      # last sentence finished -> start fresh
        context_tokens = tokenize(sentences[-1]) if sentences else []

        probs = self._next_word_probs(context_tokens)
        ranked = torch.argsort(probs, descending=True).tolist()

        results = []
        for idx in ranked:
            word = self.idx_to_word[idx]
            if idx in (PAD_IDX, UNK_IDX) or not is_word(word) or not word.startswith(partial):
                continue
            results.append({'word': word, 'score': round(probs[idx].item(), 4)})
            if len(results) == k:
                break
        return {'partial': partial, 'suggestions': results}
