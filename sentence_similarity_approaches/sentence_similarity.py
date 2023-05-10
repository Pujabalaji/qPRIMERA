import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from huggingface_hub import snapshot_download
from tensorflow_hub import KerasLayer
from transformers import AutoTokenizer, AutoModel
# from transformers import LEDConfig, LEDForConditionalGeneration
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

# print('starting to load models')

'''
PRIMER_path  = 'allenai/PRIMERA'
primera_tokenizer = AutoTokenizer.from_pretrained(PRIMER_path)
primera = LEDForConditionalGeneration.from_pretrained(PRIMER_path).cuda()
primera.gradient_checkpointing_enable()
print('loaded primera')
'''

# number of sentences to filter each doc down to
n_sentences = 5

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

def get_bert_encodings(sentences, bert_tokenizer, bert_model):
  sentence_inputs = bert_tokenizer(sentences, return_tensors='pt', padding=True)
  sentences_encoded = bert_model(sentence_inputs['input_ids'].to('cuda'))
  return sentences_encoded[0][:,0,:]

def get_filtered_documents_bert(query, documents, bert_tokenizer, bert_model):
  query_embedding = np.array(get_bert_encodings([query], bert_tokenizer, bert_model).cpu().detach())
  filtered_documents = []

  for document in documents:
    sentences = get_sentences_from_document(document)

    sentence_embeddings = []
    for sentence in sentences:
       sentence_embedding = get_bert_encodings(sentence, bert_tokenizer, bert_model).cpu().detach()[0, :]
       sentence_embeddings.append(sentence_embedding)
    sentence_embeddings = np.vstack(sentence_embeddings)

    dot_products = (sentence_embeddings @ query_embedding.T)[:, 0]
    similar_sentences = np.array(sentences)[dot_products.argsort()[-n_sentences:][::-1]]
    filtered_documents.append('. '.join(similar_sentences))
  return filtered_documents

'''
def get_summary(documents):
  input_ids = []
  for document in documents:
    document = ' '.join(document.split()[:3000])
    input_ids.extend(primera_tokenizer.encode(document))
    input_ids.append(primera_tokenizer.convert_tokens_to_ids("<doc-sep>"))
  
  input_ids = torch.tensor([input_ids]).cuda()
  output_ids = primera.generate(input_ids=input_ids, max_length=1024)
  output_str = primera_tokenizer.batch_decode(output_ids)[0]
  return output_str
'''


# cmd
# python sentence_similarity.py > filtered.txt

'''

if __name__ == "__main__": 
  
  data = read_data_from_csv('clean_text_from_urls_0.csv')

  bert_filtered_documents = []
  universal_sentence_encoder_filtered_documents = []

  for idx in range(len(data)):
    if idx % 10 == 0:
      print(f'working on idx {idx}/{len(data)}')

    query = data[idx]['query']
    docs = data[idx]['docs']

    print(f'idx {idx} has {len(docs)} docs')

    bert_filtered_documents.append('|||||'.join(get_filtered_documents_bert(query, docs)))
    universal_sentence_encoder_filtered_documents.append('|||||'.join(get_filtered_documents_universal_sentence_encoder(query, docs)))

  print('bert filtered documents')
  for doc in bert_filtered_documents:
    print(doc)

  print()

  print('universal_sentence_encoder_filtered_documents')
  for doc in universal_sentence_encoder_filtered_documents:
    print(doc)

'''