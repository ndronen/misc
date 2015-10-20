from collections import defaultdict
from operator import itemgetter
import string
import codecs
import re
import time
import itertools
import progressbar
import h5py
import json
import cPickle
import random
import Levenshtein
import enchant
from sklearn.cross_validation import train_test_split

import numpy as np
from numpy.random import RandomState

import nltk

# Data needs to be converted to input and targets.  An input is a window
# of k characters around a (possibly corrupted) token t.  A target is a
# one-hot vector representing w.  In English, the average word length
# is a little more than 5 and there are almost no words longer than 20
# characters [7].  Initially we will use a window of 100 characters.
# Since words vary in length, the token t will not be centered in the
# input.  Instead it will start at the 40th position of the window.
# This will (?) make the task easier to learn.
#
# [1] http://www.ravi.io/language-word-lengths

def load_data(path):
    with codecs.open(path, encoding='utf-8') as f:
        return f.read().replace('\n', ' ')

stanford_jar_path = '/work/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2.jar'

def build_tokenizer(tokenizer='stanford'):
    if tokenizer == 'stanford':
        return nltk.tokenize.StanfordTokenizer(
                path_to_jar=stanford_jar_path)
    else:
        return nltk.tokenize.TreebankWordTokenizer()

def is_word(token):
    return re.match(r'[\w.-]{2,}$', token)

def insert_characters(token, index_to_char, n=1, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    new_token = token
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

    new_token = token
    while new_token == token:
        idx = max(1, rng.randint(len(token)))
        ch = index_to_char[rng.randint(len(index_to_char))]
        new_token = unicode(token[0:idx-1] + ch + token[idx:])
    return new_token

def transpose_characters(token, index_to_char, n=1, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    chars = set(token)
    if len(chars) == 1:
        return token

    new_token = token
    while new_token == token:
        idx = max(1, rng.randint(len(token)))
        neighbor = 0
        if idx == 0:
            neighbor == 1
        elif idx == len(token) - 1:
            neighbor = len(token) - 2
        else:
            if rng.uniform() > 0.5:
                neighbor = idx + 1
            else:
                neighbor = idx - 1
        left = min(idx, neighbor) 
        right = max(idx, neighbor)
        new_token = unicode(token[0:left] + token[right] + token[left] + token[right+1:])
    return new_token

def tokenize(data, tokenizer=None):
    """
    Tokenize a string using a given tokenizer.

    Parameters 
    -----------
    data : str or unicode
        The string to be tokenized.
    tokenizer : str
        The name of the tokenizer.  Uses Stanford Core NLP tokenizer if
        'stanford'; otherwise, uses the Penn Treebank tokenizer.

    Returns
    ---------
    tokens : list
        An on-order list of the tokens.
    """
    toker = build_tokenizer(tokenizer=tokenizer)
    return [t.lower() for t in toker.tokenize(data)]


def build_index(token_seq, min_freq=100, max_features=1000, downsample=0):
    """
    Builds character and term indexes from a sequence of tokens.

    Parameters
    -----------
    token_seq : list
        A list of tokens from some text.
    min_freq : int
        The minimum number of occurrences a term must have to be included
        in the index.
    max_features : int
        The maximum number of terms to include in the term index.
    downsample : int
        The maximum number of occurrences to allow for any term.  Only
        used if > 0.
    """
    passes = 4
    if downsample > 0:
        passes += 1

    term_vocab = set()
    char_vocab = set([u' '])
    term_freqs = defaultdict(int)
    token_seq_index = defaultdict(list)
    below_min_freq = set()

    # Include most of the characters we care about.  We add any remaining
    # characters after this loop.
    for charlist in [string.ascii_letters, string.punctuation, range(10)]:
        for ch in charlist:
            char_vocab.add(unicode(str(ch)))

    pass_num = 1
    print('pass {pass_num} of {passes}: scanning tokens'.format(
        pass_num=pass_num, passes=passes))
    pass_num += 1
    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(token_seq)).start()

    for i, token in enumerate(token_seq):
        pbar.update(i+1)

        if not is_word(token) or len(token) == 1:
            continue

        if term_freqs[token] == 0:
            below_min_freq.add(token)
        elif term_freqs[token] > min_freq:
            try:
                below_min_freq.remove(token)
            except KeyError:
                pass

        term_freqs[token] += 1
        token_seq_index[token].append(i)

        for ch in token:
            char_vocab.add(unicode(ch))

    pbar.finish()

    print('# of terms: ' + str(len(term_freqs)))

    print('pass {pass_num} of {passes}: removing infrequent terms'.format(
        pass_num=pass_num, passes=passes))
    pass_num += 1
    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(term_freqs)).start()

    for i, term in enumerate(below_min_freq):
        pbar.update(i+1)
        del term_freqs[term]
        del token_seq_index[term]

    pbar.finish()
    print('# of terms: ' + str(len(term_freqs)))

    print('pass {pass_num} of {passes}: sorting terms by frequency'.format(
        pass_num=pass_num, passes=passes))
    pass_num += 1
    most_to_least_freq = sorted(term_freqs.iteritems(),
            key=itemgetter(1), reverse=True)
    print('')

    term_i = 0
    term_to_index = {}
    index_to_term = {}

    print('pass {pass_num} of {passes}: building term index'.format(
        pass_num=pass_num, passes=passes))
    pass_num += 1
    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(most_to_least_freq)).start()

    for i, (term, freq) in enumerate(most_to_least_freq):
        pbar.update(i+1)

        if len(term_vocab) == max_features:
            del token_seq_index[term]
            continue

        term_vocab.add(term)
        term_to_index[term] = term_i
        index_to_term[term_i] = term 
        term_i += 1

    pbar.finish()

    if downsample > 0:
        print('pass {pass_num} of {passes}: downsampling'.format(
            pass_num=pass_num, passes=passes))
        pass_num += 1
        pbar = progressbar.ProgressBar(term_width=40,
            widgets=[' ', progressbar.Percentage(),
            ' ', progressbar.ETA()],
            maxval=len(token_seq_index)).start()
        rng = random.Random(17)
        for i, token in enumerate(token_seq_index.keys()):
            pbar.update(i+1)
            indices = token_seq_index[token]
            token_seq_index[token] = random.sample(indices, downsample)
        pbar.finish()

    char_to_index = dict((c,i+1) for i,c in enumerate(char_vocab))
    char_to_index['NONCE'] = 0

    return token_seq_index, char_to_index, term_to_index

def min_dictionary_edit_distance(tokens, dictionary):
    """
    Find the edit distance from each of a list of tokens to the nearest
    word in the dictionary (where the dictionary presumably defines
    nearness as edit distance).

    Parameters
    -----------
    tokens : list
        A list of tokens.
    dictionary : enchant.Dict
        A dictionary.

    Returns 
    ----------
    distance : dict
        A dictionary with tokens as keys and (distance,token) as values.
    """
    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(tokens)).start()

    distances = []
    for i, t in enumerate(tokens):
        pbar.update(i+1)
        suggestions = [s.lower() for s in dictionary.suggest(t)]
        # If the token itself is the top suggestion, then compute
        # the edit distance to the next suggestion.  We should not
        # return an edit distance of 0 for any token.
        suggestion = suggestions[0]
        while suggestion == t:
            suggestions.pop(0)
            suggestion = suggestions[0]
            if ' ' in suggestion or '-' in suggestion:
                # For this study, we don't want to accept suggestions
                # that split a word (e.g. 'antelope' -> 'ant elope'.
                suggestion = t
            if suggestion == t + 's':
                # This is probably the plural of t.  Since the edit
                # distance of most nouns to their plural is 1, exclude
                # it.
                suggestion = t
            
        distance = (Levenshtein.distance(t, suggestion), suggestion)
        distances.append(distance)

    pbar.finish()
    assert min([d[0] for d in distances]) == 1
    return dict(zip(tokens, distances))

def build_contrasting_cases_dataset(token_seq, token_seq_index, term_to_index, char_to_index, window_size=100, token_pos=40, seed=17):
    """
    Build a dataset of examples and corresponding targets for training
    a supervised model.  Each example consists of a window around each
    token in `token_seq`.  A window consists of the leading and trailing
    characters around the token that are available to fill the window 
    up to `window_size` characters.  The token is placed in the window
    at position `token_pos`.  Each target is the index from `term_to_index`
    for the word at `token_pos`.

    Parameters
    ------------
    token_seq : list 
        A sequence of tokens.
    token_seq_index : dict of list
        A mapping from token to indices of occurrences of the token.
    term_to_index : dict
        A mapping from token to the index of the token in the vocabulary.
    char_to_index : dict
        A mapping from character to the index of the character in the vocabulary.
    window_size : int
        The size of the window for each token; includes the length of the token itself.
    token_pos : int
        The position at which the token should be placed in the window.

    Returns
    ---------
    X : np.ndarray
        A matrix with even-numbered rows containing examples without
        a known spelling error and odd-numbered rows containing a
        deliberately-injected spelling error.
    y : np.ndarray
        An array consisting of the indices in the token vocabulary of
        the correct word for each example.
    error_type : list
        A list of the type of error injected into the token.
    """
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed=seed)

    maxval = sum([len(token_seq_index[t]) for t in token_seq_index.keys()])
    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=maxval).start()

    X = []
    y = []
    error_type = []
    n = 0

    index_to_char = dict((i,c) for c,i in char_to_index.iteritems())

    for token in token_seq_index.keys():
        for i in token_seq_index[token]:
            pbar.update(n+1)

            ok_window = build_window(token_seq, i, token,
                    window_size=window_size, token_pos=token_pos)
            X.append([char_to_index[ch] for ch in ok_window])
            y.append(term_to_index[token])
            error_type.append('none')

            corruptor = rng.choice([insert_characters, delete_characters,
                replace_characters])
            corrupt_token = corruptor(token, index_to_char, n=1, seed=rng)
            corrupt_window = build_window(token_seq, i, corrupt_token,
                    window_size=window_size, token_pos=token_pos)
            X.append([char_to_index[ch] for ch in corrupt_window])
            y.append(term_to_index[token])
            error_type.append(corruptor.__name__)

            n += 1

    pbar.finish()

    return np.array(X), np.array(y)

def build_error_token_dataset(token_seq_index, term_to_index, char_to_index, n_errors_per_token=10, seed=17):
    """

    Parameters
    -----------

    Returns
    ----------

    """
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed=seed)

    n_examples = n_errors_per_token * len(token_seq_index)
    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=n_examples).start()

    corrupt_tokens = []
    targets = []
    error_type = []
    n = 0

    index_to_char = dict((i,c) for c,i in char_to_index.iteritems())

    longest = max([len(t) for t in token_seq_index.keys()]) + 1
    corrupt_tokens = np.zeros((n_examples, longest), dtype=int)
    targets = np.zeros(n_examples, dtype=int)

    for token in token_seq_index.keys():
        for i in np.arange(n_errors_per_token):
            corruptor = rng.choice([insert_characters, delete_characters,
                replace_characters, transpose_characters])
            corrupt_token = corruptor(token, index_to_char, n=1, seed=rng)
            for i, ch in enumerate(corrupt_token):
                corrupt_tokens[n, i] = char_to_index[ch]
            targets[n] = term_to_index[token]
            error_type.append(corruptor.__name__)

            n += 1
            pbar.update(n)


    pbar.finish()

    return corrupt_tokens, targets, error_type

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

def save_data(X, y, train_size, valid_size, output_prefix='data-', index_to_term=None, index_to_char=None):
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

    if index_to_term is not None:
        indices['term'] = index_to_term

        tokens = index_to_term.values()
        min_distance = defaultdict(lambda: np.inf)
        nearest_token = {}

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

        indices['min_distance'] = dict(min_distance)
        indices['min_distance_ordered'] = min_distance_ordered
        indices['nearest_token'] = nearest_token

        names = sorted(index_to_term.iteritems(), key=itemgetter(0))
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
