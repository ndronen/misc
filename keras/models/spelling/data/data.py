import re
import time
import itertools
import nltk

import numpy as np
from numpy.random import RandomState
from sklearn.preprocessing import OneHotEncoder

def load_data(path):
    with open(path) as f:
        return f.read().replace('\n', ' ')

stanford_jar_path = '/work/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2.jar'

def build_tokenizer(stanford_jar_path=stanford_jar_path):
    return nltk.tokenize.StanfordTokenizer(
            path_to_jar=stanford_jar_path)

def is_word(token):
    return re.match(r'\w+$', token)

def insert_characters(word, index_to_char, n=1, seed=1):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    idx = rng.randint(len(word))
    ch = index_to_char[rng.randint(len(index_to_char))]
    new_word = unicode(word[0:idx] + ch + word[idx:])
    return new_word

def delete_characters(word, index_to_char, n=1, seed=1):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    idx = max(1, rng.randint(len(word)))
    new_word = unicode(word[0:idx-1] + word[idx:])
    return new_word

def replace_characters(word, index_to_char, n=1, seed=1):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    idx = max(1, rng.randint(len(word)))
    ch = index_to_char[rng.randint(len(index_to_char))]
    new_word = unicode(word[0:idx-1] + ch + word[idx:])
    return new_word

# Data needs to be converted to input and targets.  An input is a window
# of k characters around a (possibly corrupted) word w.  A target is a
# one-hot vector representing w.  In English, the average word length
# is a little more than 5 and there are almost no words longer than 20
# characters [1].  Initially we will use a window of 100 characters.
# Since words vary in length, the word w will not be centered in the
# input.  Instead it will start at the 40th position of the window.
# This will make the task easier to learn.
#
# [1] http://www.ravi.io/language-word-lengths

def build_X_y(data, window_size=100, word_pos=40):
    tokenizer = build_tokenizer()
    tokens = tokenizer.tokenize(data)

    word_vocab = set(tokens)
    word_to_index = dict((w,i) for i,w in enumerate(word_vocab))
    index_to_word = dict((i,w) for i,w in enumerate(word_vocab))

    char_vocab = set([' '])
    for word in word_vocab:
        for ch in word:
            char_vocab.add(ch)
    char_to_index = dict((w,i) for i,w in enumerate(char_vocab))
    index_to_char = dict((i,w) for i,w in enumerate(char_vocab))

    X = []
    y = []

    rng = RandomState(seed=1)

    # Build a window around each word.  The surrounding context consists
    # of the leading and trailing characters that are available to fill
    # the window.
    for i, word in enumerate(tokens):

        if is_word(word) and i % 2 == 0:
            corruptor = rng.choice([insert_characters, delete_characters,
                    replace_characters])
            input_word = corruptor(word, index_to_char, n=1, seed=rng)
        else:
            input_word = word

        leading = leading_context(tokens, i, n=word_pos)
        # Subtract 2 for spaces between leading, word, and trailing.
        n = window_size - word_pos - len(input_word) - 2
        trailing = trailing_context(tokens, i, n)
        window = leading + ' ' + input_word + ' ' + trailing

        X.append([char_to_index[ch] for ch in window])
        y.append(word_to_index[word])

    X = np.array(X)
    encoder = OneHotEncoder()
    y = np.array(y).reshape((len(y), 1))
    y_one_hot = encoder.fit_transform(y)

    return X, y, y_one_hot, index_to_word, index_to_char

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
