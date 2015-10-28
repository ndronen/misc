import load_model
import rank
import cPickle

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
