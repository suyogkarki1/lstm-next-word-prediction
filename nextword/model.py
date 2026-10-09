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
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3, tie_weights=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        if tie_weights:
            # input and output word vectors share weights (Press & Wolf, 2017): fewer parameters, better LM
            assert embed_dim == hidden_dim, 'tie_weights needs embed_dim == hidden_dim'
            self.fc.weight = self.embedding.weight

    def forward(self, X):
        """Next-word logits after the last real token of each left-padded sequence: (batch, vocab).

        Padding is skipped entirely (packed sequences), so the LSTM only ever sees real words.
        An empty context (start of a sentence) runs one zero input step."""
        T = X.size(1)
        lengths = (X != PAD_IDX).sum(1)
        # left-padded -> right-padded, which is the layout pack_padded_sequence expects
        shift = (torch.arange(T, device=X.device)[None, :] + (T - lengths)[:, None]) % T
        X = X.gather(1, shift)
        embedded = self.embedding(X) * (X != PAD_IDX).unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(self.dropout(embedded), lengths.clamp(min=1).cpu(),
                                                   batch_first=True, enforce_sorted=False)
        intermediate_hidden_state, (final_hidden_state, final_cell_state) = self.lstm(packed)
        # hidden state of the top LSTM layer after the last real token
        return self.fc(self.dropout(final_hidden_state[-1]))

    def forward_all(self, X, state=None):
        """Next-word logits at every position, used for pretraining: (batch, seq, vocab)."""
        output, state = self.lstm(self.dropout(self.embedding(X)), state)
        return self.fc(self.dropout(output)), state


def expand_vocab(model, old_vocab, new_vocab, config):
    """Copy a trained model into a bigger vocabulary. Known words keep their trained
    embedding / output weights; new words start from small random vectors."""
    new_model = LSTM(len(new_vocab), **config)
    old_state, new_state = model.state_dict(), new_model.state_dict()
    for name, tensor in old_state.items():
        if new_state[name].shape == tensor.shape:
            new_state[name] = tensor.clone()
    shared = [(i, old_vocab[w]) for w, i in new_vocab.items() if w in old_vocab]
    new_ids = torch.tensor([n for n, _ in shared])
    old_ids = torch.tensor([o for _, o in shared])
    for name in ('embedding.weight', 'fc.weight', 'fc.bias'):
        new_state[name][new_ids] = old_state[name][old_ids]
    new_model.load_state_dict(new_state)
    return new_model


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

    @staticmethod
    def _sentence_tokens(text):
        """Tokens of the sentence being typed; models were trained one sentence at a time."""
        if not text.strip() or re.search(r'[.!?]\s*$', text):
            return []                           # nothing typed, or last sentence finished
        return tokenize(sent_tokenize(text.strip())[-1])

    def generate(self, text, max_words=8):
        """Greedy continuation of the current sentence.
        Returns (words, finished): finished is True if the model chose to end the sentence."""
        tokens = self._sentence_tokens(text)
        seen = set(tokens)
        words = []
        for _ in range(max_words):
            probs = self._next_word_probs(tokens)
            for idx in torch.argsort(probs, descending=True).tolist():
                word = self.idx_to_word[idx]
                # skip specials and stray punctuation; don't reuse content words (greedy decoding loops)
                if idx in (PAD_IDX, UNK_IDX) or (len(word) > 3 and word in seen) or word in words[-2:]:
                    continue
                if is_word(word) or word in ('.', '!', '?'):
                    break
            if word in ('.', '!', '?'):
                return words, True
            words.append(word)
            tokens.append(word)
            seen.add(word)
        return words, False

    def suggest(self, text, k=3):
        """Top-k suggestions. If the text ends mid-word, complete that word instead."""
        partial = ''
        context = text
        if text and not text[-1].isspace():
            match = re.search(r'(\S+)$', text)
            partial = clean(match.group(1)).strip()
            context = text[:match.start()]

        probs = self._next_word_probs(self._sentence_tokens(context))
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
