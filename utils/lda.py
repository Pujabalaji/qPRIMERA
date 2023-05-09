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

def lda_preprocess(batch_num, sent_doc=False, num_topics=20, alpha="asymmetric", beta="auto", score_filt=0.95, min_docs=3, top_n=None):
    """
    Uses LDA to select documents from a csv file.

    Arguments:
        * batch_num: the file path to the input batch csv
        * sent_doc: whether to treat each individual sentence as a distinct document
        * alpha (default auto): alpha parameter, a lower value indicates fewer topics per document
        * beta (default auto): beta parameter, a lower value indicates fewer words per topic
        * score_filt (default 0.9): filters for all documents with a similarity greater than it
        * min_docs (default 3): must include at least min_doc docs.
        * top_n (default None): if specified, selects the top n documents directly, 
                ignoring score_filt

    Returns: the list of pruned documents.

    """

    data = read_data_from_csv(batch_num)
    
    pruned_docs = []
    for datum in tqdm(data):
        query, sample = datum["query"], datum["docs"]
        
        print(query)
        docs = []
        
        if sent_doc:
            for doc in sample:
                for sent in tokenize.sent_tokenize(doc):
                    docs.append(sent)
        else:
            docs = sample

        tokens = [tokenizer(doc)['input_ids'] for doc in docs]
        query = tokenizer(query)['input_ids']

        corpus = [list(Counter(doc).items()) for doc in (tokens + [query])]
        # query_bow = list(Counter(query).items())

        # Train the model on the corpus.
        lda = LdaModel(corpus, num_topics=num_topics, alpha=alpha, eta=beta) # for some reason gensim called it eta instead of beta


        rows = []
        cols = []
        probs = []

        for i, doc in enumerate(corpus):
            for topic, prob in lda[doc]:
                rows.append(i)
                cols.append(topic)
                probs.append(prob)

        sparse = csr_matrix((probs, (rows, cols)), shape=(len(corpus), 100))
        sims = cosine_similarity(sparse)[-1,:-1]
        # print(sims)

        if top_n is not None:
            top_n_doc_indices = sorted(np.argpartition(sims, -top_n)[-top_n:])
            res = [docs[i] for i in top_n_doc_indices]
        else:
            res = [docs[i] for i in range(len(sims)) if sims[i] > score_filt]

        if len(res) < min_docs:
            actual_min = min(min_docs, len(docs))
            top_n_doc_indices = sorted(np.argpartition(sims, -actual_min)[-actual_min:])
            res = [docs[i] for i in top_n_doc_indices]

        # assert len(res) >= min(min_docs, len(docs))
        pruned_docs.append(" ||||| ".join(res))
        print(pruned_docs[-1])
    
    return pruned_docs

if __name__ == '__main__':
    lda_preprocess(0, sent_doc=True)