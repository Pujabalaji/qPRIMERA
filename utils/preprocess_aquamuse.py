# imports for web scraping in get_text_from_web() and get_common_crawl_text_with_bs4()
import requests
from bs4 import BeautifulSoup

def get_text_from_web(input_urls):
    documents = []

    for url in input_urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        documents.append(text)

    # print(documents)
    return documents

def get_common_crawl_text_with_bs4(input_urls):
    # specify the URL of the June 2018 Common Crawl index according to AQuaMuSe
    url = "https://index.commoncrawl.org/CC-MAIN-2018-22-index"

    response = requests.get(url)

    for url in input_urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            print(text)
        except:
            print("Error retrieving data from", url)