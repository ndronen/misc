from modeling.load_model import load_model
import cPickle
from operator import itemgetter
import numpy as np

MODEL_DIR = 'models/spelling/convnet/8d1c58f0737b11e5921d22000aec9897/'
INDEX_FILE = 'models/spelling/data/wikipedia-index.pkl'

def runspell(model_dir=MODEL_DIR, index_file=INDEX_FILE, k=5):
    index = cPickle.load(open(index_file))
    spell = load_model.load_model(model_dir)
    probs = spell.model.predict_proba(spell.data)
    return index, spell, probs
    #ranks = rank.compute_ranks(probs, spell.target)
    #top_k = rank.compute_top_k(probs, index['term'], index, k=k)
    #return index, spell, probs, ranks, top_k

def compute_ranks(probs, target):
    '''
    Compute the rank in a model's softmax output of each example.

    Parameters
    -----------
    probs : np.ndarray
        A 2-d array of softmax outputs with one example per row and one
        column per class.
    target : np.ndarray
        A 1-d array of target values.

    Return
    -----------
    ranks : np.ndarray
        A 1-d array of ranks, with 0 being the highest.
    '''
    ranks = np.zeros_like(target)
    for i in np.arange(len(probs)):
        idx = np.where(np.argsort(probs[i]) == target[i])[0]
        ranks[i] = probs.shape[1] - 1 - idx
    return ranks

def compute_top_k(probs, target, term_index, k=5):
    '''
    Compute the K most probable terms for each example in a model's
    softmax output.

    Parameters
    -----------
    probs : np.ndarray
        A 2-d array of softmax outputs with one example per row and one
        column per class.
    target : np.ndarray
        A 1-d array of target values.
    term_index : dict
        A mapping from vocabulary indices to vocabulary terms.

    Returns
    ----------
    top_k : dict of list of (string, list)
        A mapping from vocabulary terms to a list of lists of the K
        most probable terms and their probabilities.  The inner list has
        one entry for each row in `probs` (equivalently, each entry in
        `target`).  The outer list has one entry per row in `probs`.
    '''
    top_k = defaultdict(list)
    for i in np.arange(len(probs)):
        top_k_tokens = []
        for idx in np.argsort(probs[i])[-k:]:
            try:
                top_k_tokens.append((term_index[idx], probs[i][idx]))
            except KeyError as e:
                print('Unknown key at prediction {i} using prediction index {idx}'.format(
                    i=i, idx=idx))
        top_k_tokens = sorted(top_k_tokens, key=itemgetter(1), reverse=True)
        top_k[term_index[target[i]]].append(top_k_tokens)
    return top_k
