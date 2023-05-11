import numpy as np
import csv

def read_data_from_csv(csv_file_path):
    data = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            query = row[0]
            target = row[1]
            docs = list(filter(lambda x: x, row[2:]))
            data.append({
                "query": query,
                "docs": docs,
                "target": target
            })
    return data

# number of sentences to filter each doc down to
n_sentences = 3

def split(txt, seps):
    default_sep = seps[0]

    # we skip seps[0] because that's the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]

def get_sentences_from_document(document):
  return split(document, '.?!')

def get_filtered_documents_universal_sentence_encoder(query, documents, universal_sentence_encoder):
  query_embedding = np.array(universal_sentence_encoder([query]))
  filtered_documents = []

  for document in documents:
    sentences = get_sentences_from_document(document)
    sentence_embeddings = np.array(universal_sentence_encoder(sentences))
    dot_products = (sentence_embeddings @ query_embedding.T)[:, 0]
    similar_sentences = np.array(sentences)[dot_products.argsort()[-n_sentences:][::-1]]
    filtered_documents.append('. '.join(similar_sentences))
  return filtered_documents