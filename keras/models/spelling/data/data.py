from collections import defaultdict
from operator import itemgetter
import string
import re
import time
import itertools
import nltk
import progressbar

import numpy as np
from numpy.random import RandomState

def load_data(path):
    with open(path) as f:
        return f.read().replace('\n', ' ')

stanford_jar_path = '/work/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2.jar'

def build_tokenizer(stanford_jar_path=stanford_jar_path):
    return nltk.tokenize.StanfordTokenizer(
            path_to_jar=stanford_jar_path)

def is_word(token):
    return re.match(r'\w+$', token)

def insert_characters(token, index_to_char, n=1, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    idx = rng.randint(len(token))
    ch = index_to_char[rng.randint(len(index_to_char))]
    new_token = unicode(token[0:idx] + ch + token[idx:])
    return new_token

def delete_characters(token, index_to_char, n=1, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    idx = max(1, rng.randint(len(token)))
    new_token = unicode(token[0:idx-1] + token[idx:])
    return new_token

def replace_characters(token, index_to_char, n=1, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    idx = max(1, rng.randint(len(token)))
    ch = index_to_char[rng.randint(len(index_to_char))]
    new_token = unicode(token[0:idx-1] + ch + token[idx:])
    return new_token


# Data needs to be converted to input and targets.  An input is a window
# of k characters around a (possibly corrupted) token t.  A target is a
# one-hot vector representing w.  In English, the average word length
# is a little more than 5 and there are almost no words longer than 20
# characters [1].  Initially we will use a window of 100 characters.
# Since words vary in length, the token t will not be centered in the
# input.  Instead it will start at the 40th position of the window.
# This will (?) make the task easier to learn.
#
# [1] http://www.ravi.io/language-word-lengths

def build_X_y(data, window_size=100, token_pos=40, min_freq=5, max_features=1000):
    tokenizer = build_tokenizer()
    tokens = [t.lower() for t in tokenizer.tokenize(data)]

    token_vocab = set()

    # Limit vocabulary to the max_features-most frequent tokens that
    # occur at least min_freq times.
    token_freqs = defaultdict(int)
    for token in tokens:
        token_freqs[token] += 1

    most_to_least_freq = sorted(token_freqs.iteritems(),
            key=itemgetter(1), reverse=True)

    token_i = 0
    token_to_index = {}
    index_to_token = {}

    for token, freq in most_to_least_freq:
        if freq < min_freq or len(token_vocab) == max_features:
            break
        if not is_word(token):
            continue

        token_vocab.add(token)
        token_to_index[token] = token_i
        index_to_token[token_i] = token
        token_i += 1

    char_vocab = set([u' '])

    # This includes most of the characters we care about.  We add any
    # remaining characters after this loop.
    for charlist in [string.ascii_letters, string.punctuation, range(10)]:
        for ch in charlist:
            char_vocab.add(unicode(str(ch)))

    for token in tokens:
        for ch in token:
            char_vocab.add(ch)

    char_to_index = dict((c,i) for i,c in enumerate(char_vocab))
    index_to_char = dict((i,c) for i,c in enumerate(char_vocab))

    X = []
    y = []

    rng = RandomState(seed=17)

    # Build a window around each token.  The surrounding context consists
    # of the leading and trailing characters that are available to fill
    # the window.

    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(tokens)).start()

    for i, token in enumerate(tokens):
        pbar.update(i+1)

        if token not in token_to_index:
            continue

        ok_window = build_window(tokens, i, token,
                window_size=window_size, token_pos=token_pos)
        X.append([char_to_index[ch] for ch in ok_window])
        y.append(token_to_index[token])

        corruptor = rng.choice([insert_characters, delete_characters,
            replace_characters])
        corrupt_token = corruptor(token, index_to_char, n=1, seed=rng)
        corrupt_window = build_window(tokens, i, corrupt_token,
                window_size=window_size, token_pos=token_pos)
        X.append([char_to_index[ch] for ch in corrupt_window])
        y.append(token_to_index[token])

    pbar.finish()

    X = np.array(X)
    y = np.array(y)

    return X, y, index_to_token, index_to_char

def context_is_complete(tokens, n):
    context = ' '.join(tokens)
    return len(context) >= n

def trim_leading_context(tokens, n):
    context = ' '.join(tokens)
    start = len(context) - n
    return context[start:]

def trim_trailing_context(tokens, n):
    context = ' '.join(tokens)
    return context[:n]

def leading_context(tokens, i, n):
    end = max(0, i)
    available = tokens[:end]
    context = []
    while len(available):
        current = available.pop()
        context.insert(0, current)
        if context_is_complete(context, n):
            break
    if not context_is_complete(context, n):
        context.insert(0, ' ' * n)
    return trim_leading_context(context, n)

def trailing_context(tokens, i, n):
    start = min(len(tokens), i+1)
    available = tokens[start:]
    context = []
    while len(available):
        current = available.pop(0)
        context.append(current)
        if context_is_complete(context, n):
            break
    if not context_is_complete(context, n):
        context.append(' ' * n)
    return trim_trailing_context(context, n)

def build_window(tokens, i, token, window_size=100, token_pos=40):
    leading = leading_context(tokens, i, n=token_pos)
    # Subtract 2 for spaces between leading, token, and trailing.
    n = window_size - token_pos - len(token) - 2
    trailing = trailing_context(tokens, i, n)
    return leading + ' ' + token + ' ' + trailing
