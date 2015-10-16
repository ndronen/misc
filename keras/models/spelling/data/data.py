from collections import defaultdict
from operator import itemgetter
import string
import re
import time
import itertools
import nltk
import progressbar
import h5py
import json
import cPickle
import random
from sklearn.cross_validation import train_test_split

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
    return re.match(r'[\w.-]{2,}$', token)

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

def build_X_y(data, window_size=100, token_pos=40, min_freq=100, max_features=1000, downsample=True):
    tokenizer = build_tokenizer()
    tokens = [t.lower() for t in tokenizer.tokenize(data)]

    token_vocab = set()

    # Limit vocabulary to the max_features-most frequent tokens that
    # occur between min_freq and max_freq times.
    token_freqs = defaultdict(int)
    token_index = defaultdict(list)

    for i, token in enumerate(tokens):
        token_freqs[token] += 1
        token_index[token].append(i)

    most_to_least_freq = sorted(token_freqs.iteritems(),
            key=itemgetter(1), reverse=True)

    token_i = 0
    token_to_index = {}
    index_to_token = {}

    for token, freq in most_to_least_freq:
        if freq < min_freq or len(token_vocab) == max_features:
            del token_index[token]
            continue

        if not is_word(token) or len(token) == 1:
            del token_index[token]
            continue

        token_vocab.add(token)
        token_to_index[token] = token_i
        index_to_token[token_i] = token
        token_i += 1

    # Downsample a random subset of min_freq indices from the index.
    if downsample:
        rng = random.Random(17)
        for token in token_index.keys():
            indices = token_index[token]
            token_index[token] = random.sample(indices, min_freq)


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

    maxval = sum([len(token_index[t]) for t in token_index.keys()])
    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=maxval).start()

    n = 0

    for token in token_index.keys():
        for i in token_index[token]:
            pbar.update(n+1)
            n += 1

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
    end = max(0, i-1)
    j = end
    context = []
    while j > 0:
        context.insert(0, tokens[j])
        if context_is_complete(context, n):
            break
        j -= 1
    if not context_is_complete(context, n):
        context.insert(0, ' ' * n)
    return trim_leading_context(context, n)

def trailing_context(tokens, i, n):
    start = min(len(tokens), i+1)
    j = start
    context = []
    while j < len(tokens):
        context.append(tokens[j])
        j += 1
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

def save_data(X, y, train_size, valid_size, output_prefix='data-', index_to_token=None, index_to_char=None):
    if train_size % 2 != 0:
        raise ValueError('train_size ({train_size}) must be even'.format(
            train_size=train_size))
    if valid_size % 2 != 0:
        raise ValueError('valid_size ({valid_size}) must be even'.format(
            valid_size=valid_size))

    evens = np.arange(0, len(X), 2)
    train_evens, other_evens = train_test_split(evens,
            train_size=train_size/2, random_state=17)
    valid_evens, test_evens = train_test_split(other_evens,
            train_size=valid_size/2, random_state=17)

    train_i = np.concatenate([train_evens, train_evens+1])
    valid_i = np.concatenate([valid_evens, valid_evens+1])
    test_i = np.concatenate([test_evens, test_evens+1])

    X_train = X[train_i]
    X_valid = X[valid_i]
    X_test = X[test_i]

    y_train = y[train_i]
    y_valid = y[valid_i]
    y_test = y[test_i]

    f_train = h5py.File(output_prefix + 'train.h5', 'w')
    f_train.create_dataset('X', data=X_train, dtype=int)
    f_train.create_dataset('y', data=y_train, dtype=int)
    f_train.close()

    f_valid = h5py.File(output_prefix + 'valid.h5', 'w')
    f_valid.create_dataset('X', data=X_valid, dtype=int)
    f_valid.create_dataset('y', data=y_valid, dtype=int)
    f_valid.close()

    f_test = h5py.File(output_prefix + 'test.h5', 'w')
    f_test.create_dataset('X', data=X_test, dtype=int)
    f_test.create_dataset('y', data=y_test, dtype=int)
    f_test.close()

    indices = {}

    if index_to_token is not None:
        indices['token'] = index_to_token

        tokens = index_to_token.values()
        min_distance = defaultdict(lambda: np.inf)

        for i,token in enumerate(tokens):
            for j,other_token in enumerate(tokens):
                if i == j:
                    continue
                dist = Levenshtein.distance(token, other_token)
                if dist < min_distance[token]:
                    min_distance[token] = dist
                    nearest_token[token] = other_token

        min_distance_ordered = sorted(min_distance.iteritems(),
                key=itemgetter(1), reverse=True)

        indices['min_distance'] = min_distance
        indices['min_distance_ordered'] = min_distance_ordered
        indices['nearest_token'] = nearest_token

        names = sorted(index_to_token.iteritems(), key=itemgetter(0))
        names = [n[1] for n in names]
        target_data = {}
        target_data['y'] = {}
        target_data['y']['names'] = names
        target_data['y']['weights'] = dict(zip(range(len(names)), [1] * len(names)))
        json.dump(target_data, open(output_prefix + 'target-data.json', 'w'))

    if index_to_char is not None:
        indices['char'] = index_to_char

    if len(indices):
        cPickle.dump(indices, open(output_prefix + 'index.pkl', 'w'))
