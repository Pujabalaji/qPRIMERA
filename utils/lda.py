from preprocess_aquamuse import get_text_from_web
from read_batch import read_data_from_csv

from collections import Counter
from tqdm import tqdm

import numpy as np
from nltk import tokenize
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import LdaModel

from transformers import AutoTokenizer

import nltk
nltk.download('punkt')
tokenizer = AutoTokenizer.from_pretrained('./PRIMER')
print('tokenizer loaded')

def lda_preprocess(batch_num, out_csv_path=None, alpha="symmetric", beta="auto", score_filt=0.9, min_docs=5, top_n=None):
    """
    Uses LDA to select documents from a csv file.

    Arguments:
        * batch_num: the file path to the input batch csv
        * out_csv_path: the file path to the output csv
        * alpha (default 0.5): alpha parameter, a lower value indicates fewer topics per document
        * beta (default 0.1): beta parameter, a lower value indicates fewer words per topic
        * score_filt (default 0.5): filters for all documents with a similarity greater than it
        * min_docs (default 5): must include at least 5.
        * top_n (default None): if specified, selects the top n documents directly, 
                ignoring score_filt

    Returns:

    """

    samples = read_data_from_csv(batch_num)
    
    pruned_docs = []
    for sample in tqdm(samples):
        docs = []
        for doc in sample:
            for sent in tokenize.sent_tokenize(doc):
                docs.append(sent)

        tokens = [tokenizer(doc)['input_ids'] for doc in docs]
        query = tokenizer(query)['input_ids']

        corpus = [list(Counter(doc).items()) for doc in tokens]
        query_bow = list(Counter(query).items())

        # Train the model on the corpus.
        lda = LdaModel(corpus, alpha, eta=beta) # for some reason gensim called it eta instead of beta


        rows = []
        cols = []
        probs = []

        for i, doc in enumerate(corpus+[query_bow]):
            for topic, prob in lda[doc]:
                rows.append(i)
                cols.append(topic)
                probs.append(prob)

        sparse = csr_matrix((probs, (rows, cols)), shape=(len(corpus)+1, 100))
        sims = cosine_similarity(sparse)[-1,:-1]
        print(sims)

        if top_n is not None:
            top_n_doc_indices = np.argpartition(sims, -top_n)[-top_n:]
            res = [docs[i] for i in top_n_doc_indices]
        else:
            res = [docs[i] for i in range(len(sims)) if sims[i] > score_filt]

        if len(res) < min_docs:
            top_n_doc_indices = np.argpartition(sims, -min_docs)[-min_docs:]
            res = [docs[i] for i in top_n_doc_indices]

        pruned_docs.append("|||||".join(res))
        print(pruned_docs[-1])
    
    return pruned_docs

if __name__ == '__main__':
    print(lda_preprocess(0))