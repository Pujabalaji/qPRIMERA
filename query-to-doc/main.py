import tensorflow as tf
import requests
from rank_bm25 import BM25, BM25Okapi, BM25L, BM25Plus
from nltk.tokenize import sent_tokenize
import re

def compute_query_to_document_scores(query, docs):
    # create an instance of the BM25 class
    tokenized_corpus = [doc.split(" ") for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    # give BM25 the query and return the doc scores
    tokenized_query = query.split(" ")
    return bm25.get_scores(tokenized_query)

def read_file(filename):
    # Define the names and types of each feature in the TFRecord file
    feature_description = {
        'target': tf.io.FixedLenFeature([], tf.string),
        'query': tf.io.FixedLenFeature([], tf.string),
        'input_urls': tf.io.VarLenFeature(tf.string),
    }

    # Open the TFRecord file and create a dataset
    raw_dataset = tf.data.TFRecordDataset(filename)

    # Parse each example in the dataset using the feature_description
    parsed_dataset = raw_dataset.map(lambda example: tf.io.parse_single_example(example, feature_description))

    # Extract the values of each feature from the parsed dataset
    items = []
    for example in parsed_dataset:
        target = example['target'].numpy()
        query = example['query'].numpy().decode()
        input_urls = example['input_urls'].values.numpy().tolist()

        item = {"query": query, "target": target, "input_urls": input_urls}
        items.append(item)
    
    return items

def document_to_sentences(document):
    sentences = [
        s.strip()
        for p in re.split("\n+", document)
        # for s in re.split(r"\.|\?|!", p)
        for s in sent_tokenize(p)
        if s.strip() != ""
    ]
    return sentences

def get_document(url):
    # TODO: actually get document from url
    try:
        response = requests.get(url)
        print(response.status_code)
    except Exception as e:
        print(e)

def main():
    folders = ["dev", "test", "train"]
    for i in range(50):
        for folder in folders:
            filename = "./aquamuse_v3/v3/extractive/" + folder + "/release-set-tf-examples-0000" + str(i) + "-of-00050"
            dataset = read_file(filename)
            for data in dataset:
                query = data["query"].decode('ascii')
                urls = data["input_urls"][0].decode('ascii').split("<EOD>")
                all_sentences = []
                for url in urls:
                    document = get_document(url)
                    sentences = document_to_sentences(document)
                    all_sentences.append(sentences)
                sentences_ranked = compute_query_to_document_scores(query, all_sentences)
                target = data["target"].decode('ascii')
                # TODO: check ranked sentences against target extractive summary and score how good the ranking algorithm was

if __name__ == "__main__":
    main()