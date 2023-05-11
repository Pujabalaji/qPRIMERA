import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sentence_similarity

import evaluate
from csv import DictWriter

from transformers import (
    AutoTokenizer,
    LEDForConditionalGeneration,
    AutoModel,
)
import torch
from datasets import Dataset, load_dataset, load_metric
import statistics

from huggingface_hub import snapshot_download
from tensorflow_hub import KerasLayer

"""
Working directory: qPRIMERA/script/
Tested for 1 sample summary and prediction
Scorer: https://huggingface.co/spaces/evaluate-metric/rouge
"""
def compute_rouge_scores(predictions, references, filename):
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]

    assert len(predictions) == len(references)
    all_scores = []
    rouge = evaluate.load('rouge')

    my_file = open(filename, 'r+')
    FIELDNAMES = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    dictwriter = DictWriter(my_file, fieldnames=FIELDNAMES)

    results = rouge.compute(predictions=predictions, references=references)
    print(results)
    all_scores.append(results)
    new_row = {'rouge1': results['rouge1'], 'rouge2': results['rouge2'], 'rougeL': results['rougeL'], 'rougeLsum': results['rougeLsum']}
    dictwriter.writerow(new_row)
    my_file.close()
    return

def process_document(documents):
    input_ids_all=[]
    for data in documents:
        all_docs = data.split("|||||")[:-1]
        for i, doc in enumerate(all_docs):
            doc = doc.replace("\n", " ")
            doc = " ".join(doc.split())
            all_docs[i] = doc

        #### concat with global attention on doc-sep
        input_ids = []
        for doc in all_docs:
            input_ids.extend(
                TOKENIZER.encode(
                    doc,
                    truncation=True,
                    max_length=4096 // len(all_docs),
                )[1:-1]
            )
            input_ids.append(DOCSEP_TOKEN_ID)
        input_ids = (
            [TOKENIZER.bos_token_id]
            + input_ids
            + [TOKENIZER.eos_token_id]
        )
        input_ids_all.append(torch.tensor(input_ids))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_all, batch_first=True, padding_value=PAD_TOKEN_ID
    )
    return input_ids.cuda()


def batch_process(batch):
    input_ids=process_document(batch['document'])
    # get the input ids and attention masks together
    global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)
    # put global attention on <s> token

    global_attention_mask[:, 0] = 1
    global_attention_mask[input_ids == DOCSEP_TOKEN_ID] = 1
    generated_ids = MODEL.generate(
        input_ids=input_ids,
        global_attention_mask=global_attention_mask,
        use_cache=True,
        max_length=1024,
        num_beams=5,
    )
    generated_str = TOKENIZER.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True
        )
    result={}
    result['generated_summaries'] = generated_str
    result['gt_summaries']=batch['summary']
    return result

def run_model(pruned_docs_all, starting_indx):
    # starting_indx = index number in aquamuse dataset (clean_text_from_urls_${starting_indx}.csv)
    # pruned_docs = list of dataset_smalls = list of samples
    # dataset_small = 1 sample = "... ||||| .... ||||| ..."
    # result_small = generated summary
    rouge = load_metric("rouge")
    filename = f'average_rouge_scores_{starting_indx}.csv'
    my_file = open(filename, 'r+')

    rouge1p_list, rouge1r_list, rouge1f_list, rouge2r_list, rouge2p_list, rouge2f_list, rougeLr_list, rougeLp_list , rougeLf_list = ([] for i in range(9))
    for indx in range(len(pruned_docs_all)):
        # sample = {'document': pruned_docs_all[indx], 'summary': dataset['validation'][indx]['target']}
        def gen():
            yield {'document': pruned_docs_all[indx], 'summary': dataset['validation'][starting_indx + indx]['target']}
        dataset_small = Dataset.from_generator(gen)
        print(dataset_small)
        result_small = dataset_small.map(batch_process, batched=True, batch_size=2)
        print("result_small['generated_summaries']") # {'generated_summaries': ..., 'gt_summaries': ...}
        score=rouge.compute(predictions=result_small["generated_summaries"], references=result_small["gt_summaries"])
        print(score['rouge1'].mid)
        print(score['rouge2'].mid)
        print(score['rougeL'].mid)

        rouge1r_list.append(score['rouge1'].mid.recall)
        rouge1f_list.append(score['rouge1'].mid.fmeasure)
        rouge1p_list.append(score['rouge1'].mid.precision)

        rouge2r_list.append(score['rouge2'].mid.recall)
        rouge2f_list.append(score['rouge2'].mid.fmeasure)
        rouge2p_list.append(score['rouge2'].mid.precision)

        rougeLr_list.append(score['rougeL'].mid.recall)
        rougeLf_list.append(score['rougeL'].mid.fmeasure)
        rougeLp_list.append(score['rougeL'].mid.precision)
        
    avg_rouge1r = statistics.mean(rouge1r_list)
    avg_rouge1f = statistics.mean(rouge1f_list)
    avg_rouge1p = statistics.mean(rouge1p_list)

    avg_rouge2r = statistics.mean(rouge2r_list)
    avg_rouge2f = statistics.mean(rouge2f_list)
    avg_rouge2p = statistics.mean(rouge2p_list)

    avg_rougeLr = statistics.mean(rougeLr_list)
    avg_rougeLf = statistics.mean(rougeLf_list)
    avg_rougeLp = statistics.mean(rougeLp_list)

    med_rouge1r = statistics.median(rouge1r_list)
    med_rouge1f = statistics.median(rouge1f_list)
    med_rouge1p = statistics.median(rouge1p_list)

    med_rouge2r = statistics.median(rouge2r_list)
    med_rouge2f = statistics.median(rouge2f_list)
    med_rouge2p = statistics.median(rouge2p_list)

    med_rougeLr = statistics.median(rougeLr_list)
    med_rougeLf = statistics.median(rougeLf_list)
    med_rougeLp = statistics.median(rougeLp_list)

    field_names = ['avg_rouge1r', 'avg_rouge1f', 'avg_rouge1p', \
                   'avg_rouge2r', 'avg_rouge2f', 'avg_rouge2p', \
                    'avg_rougeLr', 'avg_rougeLf', 'avg_rougeLp', \
                    'med_rouge1r', 'med_rouge1f', 'med_rouge1p', \
                    'med_rouge2r', 'med_rouge2f', 'med_rouge2p', \
                    'med_roungeLr', 'med_roungeLf', 'med_roungeLp']
    dictwriter = DictWriter(my_file, fieldnames=field_names)
    new_row = {
        'avg_rouge1r': avg_rouge1r, 'avg_rouge1f': avg_rouge1f , 'avg_rouge1p': avg_rouge1p, \
        'avg_rouge2r': avg_rouge2r, 'avg_rouge2f': avg_rouge2f, 'avg_rouge2p': avg_rouge2p, \
        'avg_rougeLr': avg_rougeLr, 'avg_rougeLf': avg_rougeLf, 'avg_rougeLp': avg_rouge2p, \
        'med_rouge1r': med_rouge1r, 'med_rouge1f': med_rouge1f, 'med_rouge1p': med_rouge1p, \
        'med_rouge2r': med_rouge2r, 'med_rouge2f': med_rouge2f, 'med_rouge2p': med_rouge2p, \
        'med_roungeLr': med_rougeLr, 'med_roungeLf': med_rougeLf, 'med_roungeLp': med_rougeLp
    }
    dictwriter.writerow(new_row)
    my_file.close()
    return

if __name__ == "__main__": 
    PRIMER_path  = 'allenai/PRIMERA'
    TOKENIZER = AutoTokenizer.from_pretrained(PRIMER_path)
    MODEL = LEDForConditionalGeneration.from_pretrained(PRIMER_path).cuda()
    MODEL.gradient_checkpointing_enable()
    PAD_TOKEN_ID = TOKENIZER.pad_token_id
    DOCSEP_TOKEN_ID = TOKENIZER.convert_tokens_to_ids("<doc-sep>")

    dataset = load_dataset("aquamuse", "extractive")

    universal_sentence_encoder_path = snapshot_download(repo_id="Dimitre/universal-sentence-encoder")
    universal_sentence_encoder = KerasLayer(handle=universal_sentence_encoder_path)

    for starting_indx in 0, 50, 100, 600:
        csv_filename = f'clean_text_from_urls_{starting_indx}_final.csv'
        data = sentence_similarity.read_data_from_csv(csv_filename)
        pruned_docs_all = []
        for idx in range(len(data)):
            query = data[idx]['query']
            docs = data[idx]['docs']
            
            pruned_docs_all.append(' ||||| '.join(sentence_similarity.get_filtered_documents_universal_sentence_encoder(query, docs, universal_sentence_encoder)))
        
        run_model(pruned_docs_all, starting_indx)    
