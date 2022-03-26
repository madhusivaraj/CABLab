import pandas as pd
import numpy as np
import string
import sys
import collections
from collections import Counter
import matplotlib.pyplot as plt
import statistics
import math
import re
import os
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit
from sklearn.feature_extraction.text import CountVectorizer

def read_chapter(path):
    with open(path, "r", encoding="utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n", " ").replace("\r", "")
    return text

def getWordOccurrence(word):
    occurrences = collections.defaultdict(list)

    def find_word(w, s):
        return Counter(w.lower() for w in re.findall(r"\w+", s))

    for index, row in hp_texts.iterrows():
        book, chapter, text = row["Book No."], row["Chapter No."], row["Text"]
        wf = dict(find_word(word, row["Text"]))
        lcw = str(word).lower().replace('.', '')
        if lcw in wf:
            occurrences[book].append((chapter, wf[lcw]))
    return dict(occurrences)

def getWordCountByBook(dictionary):
    word_count_book = {}

    for book, value in dictionary.items():
        wc_book = 0
        for chapter in value:
            wc_book += chapter[1]
        word_count_book[book] = wc_book
    return word_count_book

def getTotalWordCount(df):
    word_count = []
    for index, row in df.iterrows():
        chapter = list(filter(None, df.iloc[index][2].strip().split(" ")))
        word_count.append(len(chapter))
    return sum(word_count)

def getChapterWordCount(book, chapter):
    chapter_wc = hp_texts[(hp_texts["Book No."]==book) & (hp_texts["Chapter No."]==chapter)]["Word Count"]
    return int(chapter_wc)

def getBookWordCount(book):
    book_wc = 0
    chapters_in_book = [i for i in hp_texts[(hp_texts["Book No."]==book)]["Chapter No."]]
    for chapter in chapters_in_book:
        book_wc += int(hp_texts[(hp_texts["Book No."]==book) & (hp_texts["Chapter No."]==chapter)]["Word Count"])
    return book_wc

def getLocations(word, text):
    text_dict = text.split()
    indices = [i for i, x in enumerate(text_dict) if x==word or x==word.lower() or x==word.capitalize()]
    return indices

def getAllWordLocations(word):
    word_locations = []
    for index, row in hp_texts.iterrows():
        chapter_text = hp_texts.at[index, "Text"]
        locations = getLocations(word, chapter_text)
        if locations != []:
            word_locations.append((hp_texts.at[index, "Book No."],
                                   hp_texts.at[index, "Chapter No."],
                                   locations))
    return word_locations

def findSpacingBetweenWords(word):
    def spaceBetweenWords(prev, curr):
        prev_book, prev_chapter, prev_idx = prev[0], prev[1], prev[2]
        curr_book, curr_chapter, curr_idx = curr[0], curr[1], curr[2]
        space = 0
        if prev_book == curr_book and prev_chapter == curr_chapter:
            return curr_idx - prev_idx - 1
        elif prev_book == curr_book and prev_chapter < curr_chapter:
            space += getChapterWordCount(curr_book, prev_chapter) - prev_idx # words left over in prev chapter
            for chapter in range(prev_chapter+1, curr_chapter):
                space += getChapterWordCount(curr_book, chapter)
            return space + curr_idx - 1
        elif prev_book < curr_book:
            space += getChapterWordCount(prev_book, prev_chapter) - prev_idx  # finish prev chapter of prev book
            # words left over in prev book (prev_book+1 to end of prev_book)
            chapters_left_prev = [i for i in hp_texts[(hp_texts["Book No."]==prev_book)]["Chapter No."]]
            for chapter in range(prev_chapter+1, chapters_left_prev[-1]+1):
                space += getChapterWordCount(prev_book, chapter)
            # words in books in between
            books_inbetween = list(range(prev_book+1,curr_book))
            for book in books_inbetween:
                space += getBookWordCount(book)
            # words in curr_book, up to curr_chapter
            for chap in range(1, curr_chapter):
                space += getChapterWordCount(curr_book, chap)
            # words in curr_chapter, up to curr_idx
            return space + curr_idx - 1
        return space

    locations = getAllWordLocations(word)
    expanded_locations = []
    for book, chapter, loc in locations:
        for l in loc:
            expanded_locations.append((book, chapter, l))
    if len(expanded_locations) == 0:
        # print(f"The word '{word}' never appears in the HP Series (Books 1-7).")
        return float(np.NaN)
    elif len(expanded_locations) == 1:
        # print(f"The word '{word}' appears once in the HP Series in Book {expanded_locations[0][0]}, Chapter {expanded_locations[0][1]}, Position {expanded_locations[0][2]}.")
        return float(np.NaN)
    spacing = []
    prev = expanded_locations[0][0], expanded_locations[0][1], expanded_locations[0][2]
    for book, chapter, loc in expanded_locations[1:]:
        curr = book, chapter, loc
        spacing.append(spaceBetweenWords(prev, curr))
        prev = curr
    return spacing

def calculateWF_HP(unigram_occurrences, all_word_occurrences):
    return unigram_occurrences / all_word_occurrences

def calculateWF_SUBTLEX(word, path="../../Data/WF_databases/SUBTLEXusfrequencyabove1.csv"):
    df = pd.read_csv(path)
    subtlex_words = df['Word'].to_list()
    if word in subtlex_words:
        wf_subtlex = df[df['Word'] == word]['Lg10WF'].tolist()[0]
        return wf_subtlex
    # print(f"The word '{word}' never appears in SUBTLEXus.")
    return np.nan

def calculateCD(occurrence_count, total_passage_count):
    # occurrence_count = number of passages (chapters) a given word appears in
    # total_passage_count = number of total passages (chapters)
    cd = float(occurrence_count/total_passage_count)
    return cd

def calculateAverageBurstingness(word):
    spacing = findSpacingBetweenWords(word)
    if np.isnan(spacing).all():
        return float(spacing)
    burstingness = sum(spacing)/len(spacing)
    return burstingness

if __name__ == '__main__':
    args = sys.argv[1:] # input from command line
    hp_texts = pd.DataFrame(columns=['Book No.', 'Chapter No.', 'Text', 'Word Count'])
    book_dir = "../Preprocessing/"
    ind = 0
    for i in range(1, 8):
        filepath = book_dir + "HPBook" + str(i) + "/"
        chapters = sorted([f for f in os.listdir(filepath)])
        for c in range(len(chapters)):
            text = read_chapter(filepath + chapters[c])
            wc = len(text.split())
            hp_texts.loc[ind] = [i,c+1,text,wc]
            ind+=1
    for word in args:
        occurrences = getWordOccurrence(word)
        if len(occurrences)!=0:
            wc_by_book = getWordCountByBook(occurrences)
            wf_hp = calculateWF_HP(sum(wc_by_book.values()), getTotalWordCount(hp_texts))
            wf_subtlex = calculateWF_SUBTLEX(word)
            passage_count = 0
            for key, val in occurrences.items():
                passage_count += len(val)
            cd = calculateCD(passage_count, len(hp_texts["Chapter No."].tolist()))
            br = calculateAverageBurstingness(word)
            print("\nInput Word: " + word,
                  "\nWord Frequency (HP books): " + str(wf_hp),
                  "\nLg10 Word Frequency (SUBTLEXus): " + str(wf_subtlex),
                  "\nContextual Diversity: " + str(cd),
                  "\nBurstingness: " + str(br))
            # print(word, wf_hp, wf_subtlex, br, end="\n")
        elif len(occurrences)==0:
            print("\nInput Word: " + word,
                  "\nNever appears in the HP Series (Books 1-7)")
            # print(f"'{word}' not found in HP Series (Books 1-7)")
