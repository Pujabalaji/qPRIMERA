import csv

def convert_str_to_list(string_list):
    string_list = string_list.strip("[]")
    list_items = string_list.split("\', \'")
    return [item[1:-1] for item in list_items]

def read_data_from_csv(batch=0):
    csv_file_path = f'../scraped_data/clean_text_from_urls_{batch}_final.csv'
    data = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            query, docs, target = row
            docs = convert_str_to_list(docs)
            data.append({
                "query": query,
                "docs": docs,
                "target": target
            })
    return data