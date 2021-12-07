'''
IMPORTS AND PRESETS
'''

# NLTK STUFF
import nltk
try:
    from nltk.corpus import stopwords
except:
    nltk.download('stopwords')
    nltk.download('popular')
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn

# BERT EXTRACTIVE SUMMARIZER
from summarizer import Summarizer

# KEYWORD EXTRACTION TOOLS
import pprint
import itertools
import re
import pke
import string
from flashtext import KeywordProcessor

# PSWD and Others
import requests
import json
import re
import os
import random
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import simple_lesk
from pywsd.lesk import cosine_lesk

# Helper Code
from mrac_qa_v1 import MRAC_QA
from questiongenerator import print_qa
from questiongenerator import QuestionGenerator

# Data Path
path_to_docs = '/NLP'

# Initialize Helpers
mrac = MRAC_QA()
qg = QuestionGenerator()

'''
===============================================================================================================================
'''


'''
Data Helper Functions
'''

def get_file(topic):

    _, _, docname = mrac.discord_query(topic, 5, 1)
    docs_path = []
    for i in docname:
        docs_path.append("/home/ubuntu/MRAC/MRACQAV1/NLP/" + i)
        print(i)
    
    for i in range(len(docs_path)):

        path = docs_path[i]
        print(path)
        output = ""

        with open(path) as f:
            for x in f:
                if (not x.strip().replace("\n", "")):
                    output = output + x
                    continue
                newline = x.strip()+'\n'
                if len(newline.rstrip()) >= 3:
                    output = output + newline

        output = re.sub(r'\n\s*\n', '\n\n', output)
        output = output.replace("\n", " ")

    print(output)

    return output

'''
===============================================================================================================================
'''


'''
Other Functions
'''

# Defining the Model to do Extractive Summarization
model = Summarizer()

# KEYWORD EXTRACTOR
def get_nouns_multipartite(text):

    out=[]

    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text)

    pos = {'PROPN'}

    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos, stoplist=stoplist)

    #    Build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.

    extractor.candidate_weighting(alpha=1.1,
                                  threshold=0.75,
                                  method='average')
    keyphrases = extractor.get_n_best(n=20)

    for key in keyphrases:
        out.append(key[0])

    return out

# TOKENIZER
def tokenize_sentences(text):

    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]

    return sentences

# MAPPING
def get_sentences_for_keyword(keywords, sentences):

    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values

    return keyword_sentences

# Distractors from Wordnet
def get_distractors_wordnet(syn,word):

    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)

    return distractors

# WORDSENSE
def get_wordsense(sent,word):

    word= word.lower()
    
    if len(word.split())>0:
        word = word.replace(" ","_")
    
    synsets = wn.synsets(word,'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output =  adapted_lesk(sent, word, pos='n')
        lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

# Distractors from http://conceptnet.io/
def get_distractors_conceptnet(word):

    word = word.lower()
    original_word= word
    if (len(word.split())>0):
        word = word.replace(" ","_")
    distractor_list = [] 
    url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
    obj = requests.get(url).json()

    for edge in obj['edges']:
        link = edge['end']['term'] 

        url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)
                   
    return distractor_list

'''
===============================================================================================================================
'''

'''
MAIN FUNCTION
'''

def main(topic = 'SEO'):

    if (topic == 'SEO'):
        contents = '''Search engine optimization (SEO) is the process of improving the quality and quantity of website traffic to a website or a web page from search engines. SEO targets unpaid traffic rather than direct traffic or paid traffic. Unpaid traffic may originate from different kinds of searches, including image search, video search, academic search, news search, and industry-specific vertical search engines.
                    As an Internet marketing strategy, SEO considers how search engines work, the computer-programmed algorithms that dictate search engine behavior, what people search for, the actual search terms or keywords typed into search engines, and which search engines are preferred by their targeted audience. SEO is performed because a website will receive more visitors from a search engine when websites rank higher on the search engine results page. These visitors can then potentially be converted into customers.
                    The leading search engines, such as Google, Bing, and Yahoo!, use crawlers to find pages for their algorithmic search results. Pages that are linked from other search engine indexed pages do not need to be submitted because they are found automatically. The Yahoo! Directory and DMOZ, two major directories which closed in 2014 and 2017 respectively, both required manual submission and human editorial review. Google offers Google Search Console, for which an XML Sitemap feed can be created and submitted for free to ensure that all pages are found, especially pages that are not discoverable by automatically following links in addition to their URL submission console. Yahoo! formerly operated a paid submission service that guaranteed to crawl for a cost per click; however, this practice was discontinued in 2009.
                    Search engine crawlers may look at a number of different factors when crawling a site. Not every page is indexed by the search engines. The distance of pages from the root directory of a site may also be a factor in whether or not pages get crawled.
                    Today, most people are searching on Google using a mobile device. In November 2016, Google announced a major change to the way crawling websites and started to make their index mobile-first, which means the mobile version of a given website becomes the starting point for what Google includes in their index. In May 2019, Google updated the rendering engine of their crawler to be the latest version of Chromium. Google indicated that they would regularly update the Chromium rendering engine to the latest version. In December 2019, Google began updating the User-Agent string of their crawler to reflect the latest Chrome version used by their rendering service. The delay was to allow webmasters time to update their code that responded to particular bot User-Agent strings. Google ran evaluations and felt confident the impact would be minor.
                    To avoid undesirable content in the search indexes, webmasters can instruct spiders not to crawl certain files or directories through the standard robots.txt file in the root directory of the domain. Additionally, a page can be explicitly excluded from a search engine's database by using a meta tag specific to robots (usually <meta name="robots" content="noindex"> ). When a search engine visits a site, the robots.txt located in the root directory is the first file crawled. The robots.txt file is then parsed and will instruct the robot as to which pages are not to be crawled. As a search engine crawler may keep a cached copy of this file, it may on occasion crawl pages a webmaster does not wish crawled. Pages typically prevented from being crawled include login specific pages such as shopping carts and user-specific content such as search results from internal searches. In March 2007, Google warned webmasters that they should prevent indexing of internal search results because those pages are considered search spam. In 2020 Google sunsetted the standard (and open-sourced their code) and now treats it as a hint not a directive. To adequately ensure that pages are not indexed a page-level robots meta tag should be included.
                    SEO is not an appropriate strategy for every website, and other Internet marketing strategies can be more effective, such as paid advertising through pay per click (PPC) campaigns, depending on the site operator's goals. Search engine marketing (SEM) is the practice of designing, running, and optimizing search engine ad campaigns. Its difference from SEO is most simply depicted as the difference between paid and unpaid priority ranking in search results. SEM focuses on prominence more so than relevance; website developers should regard SEM with the utmost importance with consideration to visibility as most navigate to the primary listings of their search. A successful Internet marketing campaign may also depend upon building high-quality web pages to engage and persuade internet users, setting up analytics programs to enable site owners to measure results, and improving a site's conversion rate. In November 2015, Google released a full 160-page version of its Search Quality Rating Guidelines to the public. which revealed a shift in their focus towards "usefulness" and mobile local search. In recent years the mobile market has exploded, overtaking the use of desktops, as shown in StatCounter in October 2016 where they analyzed 2.5 million websites and found that 51.3% of the pages were loaded by a mobile device. Google has been one of the companies that are utilizing the popularity of mobile usage by encouraging websites to use their Google Search Console, the Mobile-Friendly Test, which allows companies to measure up their website to the search engine results and determine how user-friendly their websites are.
                    SEO may generate an adequate return on investment. However, search engines are not paid for organic search traffic, their algorithms change, and there are no guarantees of continued referrals. Due to this lack of guarantee and uncertainty, a business that relies heavily on search engine traffic can suffer major losses if the search engines stop sending visitors. Search engines can change their algorithms, impacting a website's search engine ranking, possibly resulting in a serious loss of traffic. According to Google's CEO, Eric Schmidt, in 2010, Google made over 500 algorithm changes – almost 1.5 per day. It is considered a wise business practice for website operators to liberate themselves from dependence on search engine traffic. In addition to accessibility in terms of web crawlers (addressed above), user web accessibility has become increasingly important for SEO.
                    Optimization techniques are highly tuned to the dominant search engines in the target market. The search engines' market shares vary from market to market, as does competition. In 2003, Danny Sullivan stated that Google represented about 75% of all searches. In markets outside the United States, Google's share is often larger, and Google remains the dominant search engine worldwide as of 2007. As of 2006, Google had an 85–90% market share in Germany. While there were hundreds of SEO firms in the US at that time, there were only about five in Germany. As of June 2008, the market share of Google in the UK was close to 90% according to Hitwise. That market share is achieved in a number of countries.
                    As of 2009, there are only a few large markets where Google is not the leading search engine. In most cases, when Google is not leading in a given market, it is lagging behind a local player. The most notable example markets are China, Japan, South Korea, Russia and the Czech Republic where respectively Baidu, Yahoo! Japan, Naver, Yandex and Seznam are market leaders.
                    Successful search optimization for international markets may require professional translation of web pages, registration of a domain name with a top level domain in the target market, and web hosting that provides a local IP address. Otherwise, the fundamental elements of search optimization are essentially the same, regardless of language.
                    '''
    else:
        contents = get_file(topic)

    result = model(contents, ratio = 0.9)

    summarized_text = ''.join(result)

    keywords = get_nouns_multipartite(contents) 
    filtered_keys=[]

    for keyword in keywords:
        if keyword.lower() in summarized_text.lower():
            filtered_keys.append(keyword)
        
    sentences = tokenize_sentences(summarized_text)
    keyword_sentence_mapping = get_sentences_for_keyword(filtered_keys, sentences)
        
    key_distractor_list = {}

    for keyword in keyword_sentence_mapping:
        wordsense = get_wordsense(keyword_sentence_mapping[keyword][0],keyword)
        if wordsense:
            distractors = get_distractors_wordnet(wordsense,keyword)
            if len(distractors) ==0:
                distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors
        else:
            distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors

    index = 1

    print ("#############################################################################")
    print ("NOTE::::::::  Since the algorithm might have errors along the way, wrong answer choices generated might not be correct for some questions. ")
    print ("#############################################################################\n\n")

    duplicate_checker = []
    text = ""

    for each in key_distractor_list:

        sentence = keyword_sentence_mapping[each][0]
        pattern = re.compile(each, re.IGNORECASE)
        output = pattern.sub( " _______ ", sentence)
        if (output in duplicate_checker):
            continue
        
        # WRITING TO TEXT
        text = text + "(" + str(index) + ")" + " " + output + '\n'

        choices = [each.capitalize()] + key_distractor_list[each]
        top4choices = choices[:4]
        random.shuffle(top4choices)
        optionchoices = ['A','B','C','D']
        for idx,choice in enumerate(top4choices):
            text = text + "\t" + optionchoices[idx] + ")" + " " + choice + '\n'
        # text = text + "More options: " + choices[4:20] + "\n\n"
        text = text + "\n\n"
        duplicate_checker.append(output)
        index = index + 1

    print(text)

    with open("documents/Generated_Quiz.txt", "w") as text_file:
        text_file.write(text)

    return

'''
===============================================================================================================================
'''
