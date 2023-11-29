#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:13:27 2023

@author: cs460Group
"""

import urllib.request  
import bs4
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def main():
    nltk.download('punkt')     #do these if you havent downloaded yet
    nltk.download('stopwords')

    # prep the word stem and stopwords processors
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # article URL fetcher/parser for later use
    '''
    article_url = urllib.request.urlopen('')       #example url fetching
    article = article_url.read()
    article_parsed = bs4.BeautifulSoup(article, 'html.parser')

    paragraphs = article_parsed.fina_all('p')

    article_final = ''
    for p in paragraphs:
        article_final += p.text
    '''

    # example paragraph for now
    paragraph = ("Peter and Elizabeth took a taxi to attend the night party in the city. While in the party, Elizabeth collapsed and was rushed to the hospital. Since she was diagnosed with a brain injury, the doctor told Peter to stay besides her until she gets well. Therefore, Peter stayed with her at the hospital for 3 days without leaving.")
    # split into sentences
    sentences = paragraph.split('.')

    # create list for tokenized words
    all_words = []


    for sentence in sentences:
        #tokenize sentences
        words = word_tokenize(sentence)
        # Remove stopwords
        filtered_sentence = [word for word in words if not word.lower() in stop_words]
        # add tokenized words to all_words list
        all_words.extend(filtered_sentence)


        print(sentence)

    # create dictionary for word frequency
    frequency_table = dict()

    for word in all_words:
        #find stem of each word
        word = ps.stem(word)
        if word in stop_words:
            continue
        if word in frequency_table:
            frequency_table[word] += 1
        else:
            frequency_table[word] = 1


    print(frequency_table)
    
    
    
    
    
    
    
    
    


if __name__ == "__main__":
    main()