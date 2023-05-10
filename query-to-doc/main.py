import tensorflow as tf
import requests
from rank_bm25 import BM25, BM25Okapi, BM25L, BM25Plus
from nltk.tokenize import sent_tokenize
import re
import csv
import numpy as np
import math

csv.field_size_limit(214748364)

def convert_str_to_list(string_list):
    string_list = string_list.strip("[]")
    list_items = string_list.split("', '")
    return [item[1:-1] for item in list_items]

def read_data_from_csv(csv_file_path):
    data = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            sample = {
                "query": row[0],
                "target": row[1]
            }
            docs = []
            for i in range(2, len(row)):
                if row[i] == "":
                    break
                docs.append(row[i])
            sample["docs"] = docs
            data.append(sample)
    return data

def compute_query_to_document_scores(algo, query, docs):
    tokenized_corpus = [doc.split(" ") for doc in docs]
    bm25 = algo(tokenized_corpus)
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

def preprocess(algo, alpha=0.25):
    files = [0, 50, 100, 200, 400, 500, 600]
    all_pruned_docs = []
    for file in files:
        filepath = '../clean_scraped_data/clean_text_from_urls_' + str(file) + '.csv'
        data = read_data_from_csv(filepath)
        for i in range(len(data)):
            sample = data[i]
            # Split all docs into sentences and rank them based on the query
            sentences = []
            doc_ranges = []
            for doc in sample["docs"]:
                new_sentences = document_to_sentences(doc)
                doc_ranges.append((len(sentences), len(sentences) + len(new_sentences)))
                sentences += new_sentences
            if len(sentences) == 0:
                continue
            sentences_ranked = compute_query_to_document_scores(algo, sample["query"], sentences)
            # Sort the ranked sentences in descending order and get the indices
            sorted_indices = np.argsort(sentences_ranked)[::-1]
            num_sentences_to_keep = math.floor(len(sentences) * alpha)
            sentences_to_keep = set(sorted_indices[:num_sentences_to_keep])
            pruned_docs = []
            for doc_range in doc_ranges:
                new_doc = []
                for j in range(doc_range[0], doc_range[1]):
                    if j in sentences_to_keep:
                        new_doc.append(sentences[j])
                new_doc.append('')
                pruned_docs.append(". ".join(new_doc))
            pruned_docs = ' ||||| '.join(pruned_docs)
            all_pruned_docs.append(pruned_docs)
    return all_pruned_docs

def output_docs_to_file(output):
    with open('preprocessed_docs.py', 'w', encoding='utf-8') as file:
        file.write(output)

def main():
    algos = [("BM25Okapi", BM25Okapi), ("BM25L", BM25L), ("BM25Plus", BM25Plus)]
    output = []
    for name, algo in algos:
        pruned_docs = preprocess(algo)
        new_output = name + "_docs = " + str(pruned_docs)
        output.append(new_output)
    output = "\n".join(output)
    output_docs_to_file(output)

if __name__ == "__main__":
    main()