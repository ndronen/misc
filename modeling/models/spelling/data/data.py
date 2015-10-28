from collections import defaultdict
from operator import itemgetter
import six
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
import pandas as pd

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
    for i in six.moves.range(n):
        idx = rng.randint(len(new_token))
        ch = index_to_char[rng.randint(len(index_to_char))]
        new_token = unicode(new_token[0:idx] + ch + new_token[idx:])
    return new_token

def delete_characters(token, index_to_char, n=1, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    new_token = token
    if n > len(new_token):
        n = len(new_token) - 1
    for i in six.moves.range(n):
        try:
            idx = max(1, rng.randint(len(new_token)))
            new_token = unicode(new_token[0:idx-1] + new_token[idx:])
        except ValueError, e:
            print('new_token', new_token, len(new_token))
            raise e
    return new_token

def replace_characters(token, index_to_char, n=1, seed=17):
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed)

    new_token = token
    for i in six.moves.range(n):
        idx = max(1, rng.randint(len(new_token)))
        ch = index_to_char[rng.randint(len(index_to_char))]
        new_token = unicode(new_token[0:idx-1] + ch + new_token[idx:])
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
    for i in six.moves.range(n):
        idx = max(1, rng.randint(len(new_token)))
        neighbor = 0
        if idx == 0:
            neighbor == 1
        elif idx == len(new_token) - 1:
            neighbor = len(new_token) - 2
        else:
            if rng.uniform() > 0.5:
                neighbor = idx + 1
            else:
                neighbor = idx - 1
        left = min(idx, neighbor) 
        right = max(idx, neighbor)
        new_token = unicode(new_token[0:left] + new_token[right] + new_token[left] + new_token[right+1:])
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

def min_dictionary_edit_distance(terms, dictionary, progress=False):
    """
    Find the edit distance from each of a list of terms to the nearest
    word in the dictionary (where the dictionary presumably defines
    nearness as edit distance).

    Parameters
    -----------
    terms : list
        A list of terms.
    dictionary : enchant.Dict
        A dictionary.

    Returns 
    ----------
    distances : dict
        A dictionary with terms as keys and (distance,term,rank,rejected
        suggestions) as values.
    """
    pbar = None
    if progress:
        pbar = progressbar.ProgressBar(term_width=40,
            widgets=[' ', progressbar.Percentage(),
            ' ', progressbar.ETA()],
            maxval=len(terms)).start()

    distances = {}
    for i, t in enumerate(terms):
        if progress:
            pbar.update(i+1)
        suggestions = [s.lower() for s in dictionary.suggest(t)]
        # If the token itself is the top suggestion, then compute
        # the edit distance to the next suggestion.  We should not
        # return an edit distance of 0 for any token.
        orig_suggestions = list(suggestions)
        rejected_suggestions = []
        rank = 0
        accepted_suggestion = None
        accepted_suggestion_rank = 0
        distance = np.inf

        for suggestion in suggestions:
            rank += 1
            d = Levenshtein.distance(t, suggestion)
            if suggestion == t:
                rejected_suggestions.append((suggestion, d, rank))
            elif ' ' in suggestion or '-' in suggestion or "'" in suggestion:
                # For this study, we don't want to accept suggestions
                # that split a word (e.g. 'antelope' -> 'ant elope'.
                rejected_suggestions.append((suggestion, d, rank))
            elif suggestion == t + 's' or suggestion + 's' == t:
                # Exclude singular-plural variants.
                rejected_suggestions.append((suggestion, d, rank))
            else:
                if d < distance:
                    if accepted_suggestion is not None:
                        rejected_suggestions.append((accepted_suggestion, distance, rank))
                    accepted_suggestion = suggestion
                    accepted_suggestion_rank = rank
                    distance = d
                else:
                    rejected_suggestions.append((suggestion, d, rank))

        distances[t] = {
                'accepted': accepted_suggestion,
                'distance': distance,
                'rank': accepted_suggestion_rank,
                'rejected': rejected_suggestions
                }
            
    if progress:
        pbar.finish()

    return distances

def build_dataset(token_seq, token_seq_index, term_to_index, char_to_index, max_token_length=15, leading_context_size=10, trailing_context_size=10, leading_separator='{', trailing_separator='}', n_examples_per_context=10, n_errors_per_token=[1], n_errors_per_context=[0], dictionary=enchant.Dict('en_US'), seed=17):
    """
    Build a dataset of examples of spelling erors and corresponding
    corrections for training a supervised spelling correction model.
    Each example consists of a window around a token in `token_seq`.  

    The error types are (random) insertion, (random) deletion, (random)
    replacement, and transposition.  (In the future other error types may
    be added, such as sampling characters from the token itself instead
    of randomly (for insertion or replacement) and creating errors that
    are plausible given the layout of a QWERTY keyboard.

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
    max_token_length : int
        The maximum allowed token length for an (example, correction)
        pair.  Tokens that exceed this limit are ignored and no examples
        of spelling errors of this token are created.
    leading_context_size : int
        The size of the window for each token; includes the length of the token itself.
    trailing_context_size : int
        The position at which the token should be placed in the window.
    leading_separator : str
        A string that will be placed immediately before the spelling
        error to demarcate it from the leading context.
    trailing_separator : str
        A string that will be placed immediately after the spelling
        error to demarcate it from the trailing context.
    n_examples_per_context : int
        The number of examples of errors that will be generated per context.
    n_errors_per_token : list of int
        The possible number of errors injected into a word for a training
        example.  The number of errors injected into a word is sampled from
        this list for each training example.
    n_errors_per_context : list of int
        The possible number of errors injected into the leading and trailing
        context for a training example.  The number of errors injected into
        both the leading and trailing context is sampled from this list for
        each training example.
    dictionary : enchant.Dict
        Dictionary for computing edit distance to nearest term.
    seed : int or np.random.RandomState
        The initialization for the random number generator.

    Returns
    ---------
    spelling_errors : np.ndarray
        A matrix of examples of a deliberately-injected spelling error.
    corrections : np.ndarray
        An array consisting of the indices in the token vocabulary of
        the correct word for each example.
    error_types : list
        A list of the type of error injected into the token.
    """
    if isinstance(seed, RandomState):
        rng = seed
    else:
        rng = RandomState(seed=seed)

    n_contexts = sum([len(token_seq_index[t]) for t in token_seq_index.keys() if len(t) <= max_token_length])
    n_examples = n_examples_per_context * n_contexts
    max_inserts_per_token = max(n_errors_per_token)
    max_inserts_per_context = 2*max(n_errors_per_context)
    max_chars_in_window = leading_context_size + max_token_length + trailing_context_size + len(leading_separator) + len(trailing_separator) + max_inserts_per_token + max_inserts_per_context
    # The "background" on which each context is overlain is a sequence
    # of spaces; a space represents 'no character'.
    error_examples = np.zeros((n_examples, max_chars_in_window), dtype=np.int32)
    error_examples.fill(char_to_index[' '])
    corrections = np.zeros(n_examples, dtype=np.int32)
    error_types = []
    context_ids = []
    term_lens = []
    edit_distance_to_nearest_term = []
    nearest_term = []

    print('starting to construct {n_examples} examples'.format(
            n_examples=n_examples))

    pbar = progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=n_examples).start()
    n = 0

    index_to_char = dict((i,c) for c,i in char_to_index.iteritems())
    if 63 not in index_to_char.keys():
        # A workaround for now.
        index_to_char[63] = '^'

    for token in token_seq_index.keys():
        if len(token) > max_token_length:
            continue

        term_len = len(token)
        distances = min_dictionary_edit_distance([token], dictionary)
        distance = distances[token]['distance']
        term = distances[token]['accepted']

        for i in token_seq_index[token]:
            for j in six.moves.range(n_examples_per_context):
                pbar.update(n+1)
                corruptor = rng.choice([insert_characters, delete_characters,
                        replace_characters, transpose_characters])
                n_token_errors = rng.choice(n_errors_per_token)
                corrupt_token = corruptor(token, index_to_char, n=n_token_errors, seed=rng)

                n_context_errors = rng.choice(n_errors_per_context)
                leading = leading_context(token_seq, i, leading_context_size)
                leading = corruptor(leading, index_to_char, n=n_context_errors, seed=rng)
                trailing = trailing_context(token_seq, i, trailing_context_size)
                trailing = corruptor(trailing, index_to_char, n=n_context_errors, seed=rng)

                window = ''.join([leading, leading_separator,
                        corrupt_token, trailing_separator, trailing])

                for k, ch in enumerate(window):
                    try:
                        error_examples[n, k] = char_to_index[ch]
                    except KeyError, e:
                        error_examples[n, k] = char_to_index['_']
                corrections[n] = term_to_index[token]
                error_types.append(corruptor.__name__)
                context_ids.append(i)
                edit_distance_to_nearest_term.append(distance)
                nearest_term.append(term)
                term_lens.append(term_len)

                n += 1

    pbar.finish()

    df = pd.DataFrame({
            'correction': corrections,
            'error_type': error_types,
            'context_id': context_ids,
            'edit_distance_to_nearest_term': edit_distance_to_nearest_term,
            'nearest': nearest_term,
            'term_length': term_lens
            })
    colwidth = int(np.log10(error_examples.shape[1]))+1
    colfmt = 'c{col:0' + str(colwidth) + 'd}'
    for col in np.arange(error_examples.shape[1]):
        colname = colfmt.format(col=col)
        df[colname] = error_examples[:, col]

    return df

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

def build_fixed_width_window(tokens, i, token, window_size=100, token_pos=40):
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
