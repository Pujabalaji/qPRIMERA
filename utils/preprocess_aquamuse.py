import requests
from bs4 import BeautifulSoup

def get_text_from_web(input_urls):
    documents = []

    for url in input_urls:
        # print(url)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        documents.append(text)

    # print(documents)
    return documents

def get_common_crawl_text_with_bs4(input_urls):
    # specify the URL of the June 2018 Common Crawl index according to AQuaMuSe
    cc_url = "https://index.commoncrawl.org/CC-MAIN-2018-22-index"

    for url in input_urls:
        try:
            params = {
                "url": url
            }
            response = requests.get(cc_url, params)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            print(text)
        except:
            print("Error retrieving data from", url)


if __name__ == '__main__':
    input_urls = [ "http://www.gtopcars.com/makers/hummer/2016-hummer-h3/", "http://wiki-offline.jakearchibald.com/wiki/Hummer_H3", "http://www.autocarbase.com/2013/01/hummer-h3.html", "https://carsmag.us/2017-pagani-huayra/", "http://www.gtopcars.com/makers/cadillac/2020-cadillac-escalade/", "http://www.gtopcars.com/makers/hummer/2017-hummer-h3/" ]

    # url = input_urls[0]

    print(get_text_from_web(input_urls))
    # print(get_common_crawl_text_with_bs4(input_urls))