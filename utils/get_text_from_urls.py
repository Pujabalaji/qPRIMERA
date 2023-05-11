import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
import csv
from csv import DictWriter
import re
"""
assuming working directory is scripts/
"""

def get_warc_index(input_urls):
    # specify the URL of the June 2017 Common Crawl index according to AQuaMuSe
    url = "https://index.commoncrawl.org/CC-MAIN-2017-26-index"

    for parameter in input_urls:
        try:
            response = requests.get(url + "?=" + parameter)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            # print(text)
        except:
            print("Error retrieving data from", url)

def get_text(input_urls):
    fetched_all = True
    all_text = []
    for url in input_urls:
        try: 
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Get text from all <p> tags.
            p_tags = soup.find_all('p')
            # Get the text from each of the “p” tags and strip surrounding whitespace.
            p_tags_text = [tag.get_text().strip() for tag in p_tags]
            # Filter out sentences that contain newline characters '\n' or don't contain periods.
            sentence_list = [sentence for sentence in p_tags_text if not '\n' in sentence]
            sentence_list = [sentence for sentence in sentence_list if '.' in sentence]
            # Combine list items into string.
            article = ' '.join(sentence_list)
            all_text.append(article)
            # print("Success!"+url)
        except:
            fetched_all = False
            break
    if fetched_all:
        return all_text
    return None
    
# https://www.slideshare.net/AmazonWebServices/aws-public-data-sets-how-to-stage-petabytes-of-data-for-analysis-in-aws-wps326-aws-reinvent-2018

def write_to_csv(mode):
    field_names = ['Query', 'Target', 'Doc1', 'Doc2', 'Doc3', 'Doc4', 'Doc5', 'Doc6', 'Doc7', 'Doc8', 'Doc9', 'Doc10']

    full_length = len(dataset[mode])
    for indx in range(600, full_length): # success in 5
        if indx in [0, 50, 100, 200, 300, 400, 500, 600]:
            filename = f'clean_text_from_urls_{indx}.csv'
            my_file = open(filename, 'r+')
            dictwriter = DictWriter(my_file, fieldnames=field_names)

        input_urls = dataset[mode][indx]['input_urls']
        output = get_text(input_urls)
    
        if output:
            print("Sucess retrieving all index " + str(indx))
            new_row = {'Query': dataset[mode][indx]['query'], 'Target': dataset[mode][indx]['target']}
            for i in range(1, 11):
                if i <= len(output):
                    if len(output[i-1]) == 0:
                        break # empty string scraped
                    update = {f'Doc{i}': output[i-1]}
                    new_row.update(update)
                else:
                    new_row.update({f'Doc{i}': ''})
            dictwriter.writerow(new_row)
        else:
            print("Error getting indx " + str(indx))
        
        if indx in [49, 99, 199, 299, 399, 499, 599, full_length-1]:
            my_file.close()

if __name__ == "__main__": 
    dataset = load_dataset("aquamuse", "extractive")
    mode = 'validation'
    print(dataset)

    write_to_csv(mode)

