import tensorflow as tf
import requests
from rank_bm25 import BM25, BM25Okapi, BM25L, BM25Plus
from nltk.tokenize import sent_tokenize
import re
import csv
import numpy as np
import ast

csv.field_size_limit(214748364)

def convert_str_to_list(string_list):
    string_list = string_list.strip("[]")
    list_items = string_list.split("', '")
    return [item[1:-1] for item in list_items]

def read_data_from_csv(csv_file_path):
    data = []
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        i = 0
        for row in csv_reader:
            query, docs, target = row
            if csv_file_path == "../scraped_data/clean_text_from_urls_200_final.csv" and i == 1:
                print(docs)
            docs = convert_str_to_list(docs)
            data.append({
                "query": query,
                "docs": docs,
                "target": target
            })
            i += 1
    return data

def compute_query_to_document_scores(query, docs):
    # create an instance of the BM25 class
    tokenized_corpus = [doc.split(" ") for doc in docs]
    bm25 = None
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

# def main():
#     folders = ["dev", "test", "train"]
#     for i in range(50):
#         for folder in folders:
#             filename = "./aquamuse_v3/v3/extractive/" + folder + "/release-set-tf-examples-0000" + str(i) + "-of-00050"
#             dataset = read_file(filename)
#             for data in dataset:
#                 query = data["query"].decode('ascii')
#                 urls = data["input_urls"][0].decode('ascii').split("<EOD>")
#                 all_sentences = []
#                 for url in urls:
#                     document = get_document(url)
#                     sentences = document_to_sentences(document)
#                     all_sentences.append(sentences)
#                 sentences_ranked = compute_query_to_document_scores(query, all_sentences)
#                 target = data["target"].decode('ascii')
#                 # TODO: check ranked sentences against target extractive summary and score how good the ranking algorithm was

def main():
    files = [0, 50, 100, 200, 400, 500, 600]
    total_similarity = 0
    total_samples = 0
    for file in files:
        filepath = '../scraped_data/clean_text_from_urls_' + str(file) + '_final.csv'
        data = read_data_from_csv(filepath)
        for i in range(len(data)):
            sample = data[i]
            # Split all docs into sentences and rank them based on the query
            sentences = []
            for doc in sample["docs"]:
                sentences += document_to_sentences(doc)
            if len(sentences) == 0:
                continue
            sentences_ranked = compute_query_to_document_scores(sample["query"], sentences)
            # Sort the ranked sentences in descending order and get the indices
            sorted_indices = np.argsort(sentences_ranked)[::-1]
            # Get the top sentences
            target_sentences = set(document_to_sentences(sample["target"]))
            num_sentences_in_target = len(target_sentences)
            top_sentences = set()
            i = 0
            while len(top_sentences) != num_sentences_in_target and i < len(sorted_indices):
                top_sentences.add(sentences[sorted_indices[i]])
                i += 1
            if len(top_sentences) != num_sentences_in_target:
                print(filepath, i)
                print(len(sentences), num_sentences_in_target)
            # print("TOP SENTENCES INDEX" + str(i))
            # print(top_sentences)
            # print("-----------------------")
            # print("TARGET SENTENCES INDEX" + str(i))
            # print(target_sentences)
            # Calculate the similarity between the target summary and the top ranked sentences
            common_elements = target_sentences.intersection(top_sentences)
            similarity = len(common_elements) / len(target_sentences.union(top_sentences))
            total_similarity += similarity
            total_samples += 1
    avg_similarity = total_similarity/total_samples * 100
    print("Average similarity: ", avg_similarity, "%")

if __name__ == "__main__":
    # main()
    doc = '''
    ['The page you are looking for might have been removed, had its name changed or is temporarily unavailable. ▒ Copyright  Best Brains. We use cookies to ensure that you got the best experience on our website.  I accept', 'The 4th of July was indeed a major event for it set in motion the rebellion against a corrupt form of government (then known as monarchy) which was dominated by unelected bureaucrats. The American Revolution set off a contagion that manifested in Europe with the French Revolution beginning on the\xa0July 14, 1789. The\xa0first inauguration of\xa0George Washington\xa0took place on\xa0April 30, 1789. So if we take the Cycle of Political Change from 1789, that brings us up to a rather important event of a national hero who was called a traitor and risked being killed or imprisoned for life, exactly as King George III declared. George III declared Thomas Jefferson and everyone else who signed the Declaration of Independence, which Jefferson wrote, a traitor. Snowden could only go to Russia for security for any other country would have turned him over. George III sent an entire army to personally capture Jefferson and hang him. Fortunately, he was warned that an entire army was converging on his home and he had time to flee. That hero reappeared on\xa0May 20, 2013, precisely 224 years on cue from 1789. That hero is\xa0Edward Snowden\xa0for what he revealed was not just that the United States was unconstitutionally violating every right of every American citizen; it was a worldwide cooperation among nations to hunt down their own people because they could feel the reins of power slipping from their grip. They say history produces the heroes we need at critical moments; perhaps this is true. One is hard-pressed to find so many brilliant minds coming together, as was the case in 1776, as we saw with people like Jefferson, Ben Franklin, James Madison, and even Thomas Paine whose words moved a nation along with those of Patrick Henry ▒\xa0Give Me Liberty or Give Me Death. Hopefully, there will be others who step forward over the next four years to lend a hand once again. We face a very dark new age of totalitarianism where government is crushing all rights to preserve their privileges and power.', 'Looks like something went wrong.', 'Barack Obama will be inaugurated in front of hundreds of thousands of people on Jan. 21, but here\'s a little secret: That\'s just ceremonial. He will actually be sworn in 24 hours beforehand, in what is sure to be a much smaller ceremony. That\'s because the Constitution\'s 20th Amendment specifies that a presidential term begins on Jan. 20 each year. Since that date falls on a Sunday, the big production was pushed back a day. Which means when Obama puts his hand on the Bible in front of a national TV audience and repeats after Chief Justice John Roberts, it will just be a rerun. For an event that\'s essentially etched in stone every four years, the history behind the inauguration date is surprisingly fascinating. It\'s moved up more than three months since\xa0George Washington\'s first in April 1789, but despite mind-blowing advancements in technology, there\'s still nearly a three-month gap between election day and inauguration day. And experts say that gap isn\'t likely to shrink. Our first presidential election began on Dec. 15, 1788, and voting didn▒t end until Jan. 10. Though George Washington\'s victory was a forgone conclusion, it wouldn\'t be confirmed for weeks to come. "The long struggle to ratify the Constitution delayed the selection of the Electoral College until January 1789 and the electors did not meet in their state capitals until March," said John Ferling, author of \'The Ascent of George Washington: The Hidden Political Genius of an American Icon.\' "The Senate then had to count the ballots, and that was not done until early April." Washington got word of his election on April 14, and arrived in New York on April 23. A week later, on April 30, he was inaugurated. Even though Washington wasn\'t inaugurated until April, his first term technically ended four years after the Constitution was ratified. So for Washington\'s second term -- and every inauguration for the next 144 years -- Inauguration Day was held on March 4, the anniversary of the Constitution\'s ratification (or March 5, if the 4th fell on a Sunday). Though this shaved 47 days off the interregnum (the period between the election and the inauguration), the nation would still find itself saddled with a lame duck president for months on end. When Abraham Lincoln was elected president on Nov. 6, 1860, the nation was teetering on the brink of war, and it would be four months before he could take the reins. During that time, seven states seceded, formed the Confederate States of America, and, two weeks before Lincoln\'s inauguration, selected Jefferson Davis as their own president. It would take another nation-defining crisis before anyone would do anything to address the issue. In 1932, with the Great Depression crushing the economy, Sen. George Norris, an independent from Nebraska, proposed what became known as the "Lame Duck" amendment, seeking to move up inauguration day to Jan. 20. "He was very coy during that interregnum, where Hoover was trying to get him to sign on and start wielding authority before he had authority," Clark said. "And FDR, politically, did not want to be boxed in by being seen to endorse the policies▒ to avoid the perception that (he) was in any way responsible for what was going on." Eighty years later, with the advent of the Internet and private jets, the outcome of a presidential election is usually known hours after polls close, and the president-elect can get to D.C. by noon the following day. But presidential inaugurations are still held on Jan. 20 -- exposing America to nearly three months of lame duck presidency. Bruce Ackerman, Sterling Professor of Law and Political Science at Yale University, says this is due to vagaries of the Electoral College, a system he described as "ridiculous" and "antiquated." Ackerman, who worked for Al Gore during the disputed 2000 election, says the Electoral College requires time to mitigate disputes in close elections. "You need time to resolve this," Ackerman said. "The statute at the present time requires, if the states want to have the returns unchallenged at subsequent proceedings, they have to hand them in by the middle of December." Ackerman said we should look to the rest of the world to find a system of Democratic elections that is fast and efficient. "See how the French do it," he said. "If you look at any modern democracy other than ours, just copy their system. It has all the features you\'re describing." In France, from election day to inauguration takes about 24 days, and that\'s despite often having a second round of voting in between. Think how nice it\'d be to hold the Inaugural parade on Nov. 30, when the average temperature in DC is a balmy 52. But be prepared to actually watch that parade in chilly January weather.', "The President of the United States is the elected head of state and head of government of the United States. The president leads the executive branch of the federal government and is the commander-in-chief of the United States Armed Forces. The president is indirectly elected to a four-year term by the people through an Electoral College (or by the House of Representatives, should the Electoral College fail to award an absolute majority of votes to any person). Since the office was established in 1789, 43 people have served as president. The first, George Washington, won a unanimous vote of the Electoral College. Grover Cleveland served two non-consecutive terms in office, and is counted as the nation's 22nd and 24th president. William Henry Harrison spent the shortest time in office, dying 31 days after taking office in 1841. Franklin D. Roosevelt served the longest, over twelve years, before dying early in his fourth term in 1945; he is the only president to have served more than two terms. Since the ratification of the Twenty-second Amendment to the United States Constitution in 1951, no person may be elected president more than twice, and no one who has served more than two years of a term to which someone else was elected may be elected more than once.[1] The current president is Barack Obama, and the president-elect is Donald Trump,[2] whose term of office will commence on January 20, 2017. Of the individuals elected as president, four died in office of natural causes (William Henry Harrison,[3] Zachary Taylor,[4] Warren G. Harding,[5] and Franklin D. Roosevelt), four were assassinated (Abraham Lincoln,[6] James A. Garfield,[6][7] William McKinley,[8] and John F. Kennedy), and one resigned (Richard Nixon).[9] John Tyler was the first vice president to assume the presidency intra-term, and set the precedent that a vice president who does so becomes the fully functioning  president with his own presidency, as opposed to a caretaker president. The Twenty-fifth Amendment to the Constitution put Tyler's precedent into law in 1967. It also established a mechanism by which an intra-term vacancy in the vice presidency could be filled. Richard Nixon was the first president to fill a vacancy under this Provision when he appointed Gerald Ford to the office. Later, Ford became the second to do so when he appointed Nelson Rockefeller to succeed him. Previously, an intra-term vacancy was left unfilled. Presently, there are four living former presidents. The most recent death of a former president was that of Gerald Ford (served 1974 to 1977) on December 26, 2006 (aged 93 years, 165 days). The most recently serving president to die was Ronald Reagan (served 1981 to 1989) on June 5, 2004 (aged 93 years, 120 days). Jimmy Carter currently holds the record for having the longest post-presidency of any president. Four presidents held other high U.S. federal offices after leaving the presidency. Additionally, several presidents campaigned unsuccessfully for other U.S. state or federal elective offices after leaving the presidency.", '', 'Federal Hall National Memorial, currently administered by the National Park Service, has always been a popular landmark with tourists thanks to its position on one of the most photographed intersections in New York. Who can resist that noble statue of George Washington silently meditating on the financial juggernaut of Wall Street? In 2015 Federal Hall was officially named an official American National Treasure, part of the ongoing Saving Places program by\xa0National Trust for Historic Preservation\xa0calling attention to endangered landmarks of national significance. (The following article was originally posted that year in honor.) It joins an impressive hodgepodge of local landmarks such as South Street Seaport, the Lower East Side Tenement Museum and the Whitney Studio. While this sounds like a distinction that might pique the interest of Nicolas Cage ▒ after all, he broke into Trinity Church up the street in the first National Treasure film ▒ the National Treasure program gives a boost to historic places that may be otherwise neglected or under-appreciated. When▒s the last time you were there? 1. This isn▒t the real Federal Hall\xa0The original structure was built in 1699, built by the British who used materials from\xa0the city▒s demolished north defense wall ▒ aka the wall of Wall Street ▒ to construct it.\xa0It was the center of most governmental functions, from city administration to later federal functions. 2. That Federal Hall was remodeled by a controversial architect.\xa0Pierre Charles L▒Enfant, a successful city contractor and former Continental Army engineer redesigned the structure in time for its use as the first national capital. According to David McCullough, it was the first building in America designed to exalt the national spirit, in what would come to be known as the Federal style. L▒Enfant would later work on the creation of Washington DC from Maryland swampland and be fired from that project by George Washington. 3. George Washington was first inaugurated here on April 30, 1789. The King James bible he was sworn in with ▒ property of a New York Freemason lodge ▒ is still at Federal Hall. 4. The original Federal Hall was torn down in 1812 when city administration moved to the new City Hall.\xa0 Its materials were sold off to make other buildings in the city. 5. The current Federal Hall is actually the original U.S. Custom House which opened in 1842, replacing a structure used for that purpose at 22-24 Wall Street. 6. The offices of the Custom House again moved in 1855, and the building was used as the U.S. Sub-Treasury building. In 1913 it became the first place in New York to buy the original buffalo nickel. Below: Suffrage proponents Mrs. W.L. Prendergast, Mrs. W.L. Colt, Doris Stevens, Alice Paul stop in front of Federal Hall 7. In 1918 Charlie Chaplin and Douglas Fairbanks famously drew thousands to the steps of Federal Hall to promote the sale of war bonds. Later that year doughnuts were auctioned off from its steps as a war fund-raiser. 8. In 1920 a wagon full of dynamite exploded across the street from the Sub-Treasury, killing 38 people in what is today still an unsolved mystery. 9. The Sub-Treasury had moved out by the 1930s, and the building was officially re-opened as the Federal Hall Memorial Museum in January 1940. It was inspired in part by America▒s celebration of the 150th anniversary of Washington▒s inauguration. The\xa01939-40 World▒s Fair presented a replica of the original Federal Hall even after an earlier version of Federal Hall in Bryant Park failed to attract visitors. 10. Federal Hall received a massive renovation in 2006 after the collapse of the World Trade Center in 2001 weakened the foundations of the building. Check out the official announcement at the website for the National Trust for Historic Preservation. Top image by the Wurts Brothers, taken in 1908. Courtesy Museum of the City of New York Hi! Enjoyed the Federal Hall blog post. For #9, if the building reopened as a memorial museum in 1940, wouldn▒t that have been to commemorate the 150th anniversary of Washington▒s inauguration, not the 225th? I think we would▒ve celebrated the 225th last year in 2014. Either way, great blog and podcasts. Keep up the great work! Your email address will not be published. Required fields are marked * Save my name, email, and website in this browser for the next time I comment. Looking for the latest episode of our podcasts? Listen now on iTunes to ▒The Bowery Boys▒ and ▒The First▒. Find recent podcast episodes here, and click to read more about listening options here. Find out how you can support the production of the Bowery Boys Podcast.']
    '''
    docs = convert_str_to_list(doc)
    for text in docs:
        encoded_text = str(text.encode('ascii', errors='replace'))
        print(encoded_text)