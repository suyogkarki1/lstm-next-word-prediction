"""WikiText-2 (raw) as general English for pretraining.

The parquet files come from https://huggingface.co/datasets/Salesforce/wikitext and are
downloaded on first use into data/wikitext/ (not committed to git).
"""
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from .model import PAD, UNK, split_sentences

URL = 'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'wikitext'


def _detokenize(text):
    # WikiText splits numbers/hyphens as "1 @,@ 000" and "role @-@ playing"
    text = re.sub(r' @([-,.])@ ', r'\1', text)
    return re.sub(r' ([.,;:!?\'")])', r'\1', text)


def load_sentences(split):
    """Tokenized sentences for 'train' or 'validation', cached after the first run."""
    cache = DATA_DIR / f'wikitext2-{split}.tokens.pt'
    if cache.exists():
        return torch.load(cache)
    parquet = DATA_DIR / f'wikitext2-{split}.parquet'
    if not parquet.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f'Downloading WikiText-2 {split} ...')
        urllib.request.urlretrieve(URL.format(split=split), parquet)
    paragraphs = [t for t in pd.read_parquet(parquet)['text'] if t.strip() and not t.strip().startswith('=')]
    sentences = split_sentences('\n'.join(_detokenize(p.strip()) for p in paragraphs))
    torch.save(sentences, cache)
    return sentences


def build_general_vocab(sentences, max_size):
    counts = Counter(t for s in sentences for t in s)
    vocab = {PAD: 0, UNK: 1}
    for word, _ in counts.most_common(max_size - len(vocab)):
        vocab[word] = len(vocab)
    return vocab
