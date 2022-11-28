import csv
from globals import *
from data import get_gpt3_response, read_space_data

import os
import sys
import pickle
import json
import math
import re

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import numpy as np
import matplotlib.pyplot as plt

def sublist(list1, list2):
    l1 = len(list1)
    l2 = len(list2)
    if l2 < l1:
        return False
    for i in range(0,l2-l1+1):
        here = True
        for j in range(l1):
            if list2[i+j] != list1[j]:
                here = False
                break
        if here:
            return True
    return False

def abstractiveness(summary_sentences, reference_sentences, n=3):
    if type(summary_sentences) == str:
        summary_sentences = sent_tokenize(summary_sentences)
    if type(reference_sentences) == str:
        reference_sentences = sent_tokenize(reference_sentences)
    summary_sentences = [word_tokenize(ss) for ss in summary_sentences]
    reference_sentences = [word_tokenize(rs) for rs in reference_sentences]
    from_ref = 0
    not_from_ref = 0
    for ss in summary_sentences:
        if len(ss) < n:
            continue
        for i in range(len(ss)+1-n):
            check = ss[i:i+n]
            is_from_ref = False
            for rs in reference_sentences:
                if sublist(check, rs):
                    is_from_ref = True
                    break
            if is_from_ref:
                from_ref += 1
            else:
                not_from_ref += 1
    if from_ref + not_from_ref > 0:
        not_from_ref /= (from_ref+not_from_ref)
    return not_from_ref

def get_aspectwise_statements(topicwise_annotated, have_all_statements=False):
    aspectwise = {}
    for eid in topicwise_annotated:
        statements = {aspect : [] for aspect in aspects_[1:]}
        topicwise_eid = topicwise_annotated[eid]
        for (topics, aspects, statement) in topicwise_eid:
            if have_all_statements:
                for aspect in aspects_[1:]:
                    statements[aspect].append(statement)
            else:
                for aspect in aspects:
                    if aspect != "none":
                        statements[aspect].append(statement)
        aspectwise[eid] = statements
    return aspectwise 

def average_abstractiveness(space, summarized, n=3):
    avg_abstractiveness = 0.0
    nl = 0
    for eid in summarized:
        refs = []
        for review in space[eid]['reviews']:
            refs += review['sentences']
        for aspect in summarized[eid]:
            sents = sent_tokenize(summarized[eid][aspect])
            avg_abstractiveness += abstractiveness(sents, refs, n=n)
            nl += 1
    return avg_abstractiveness / nl       

if __name__ == '__main__':
    space = read_space_data()
    space = {entity['entity_id'] : entity for entity in space}
    pkl_names = ['tcg.pkl', 'tqg.pkl', 'qg.pkl', 'acesum.pkl', 'qfsumm.pkl', 'rg.pkl']
    sum_dir = os.path.join(SPACE_SAVE_DATA_ROOT, "all-new-pkls", "summaries-pkl")
    sr_dir = os.path.join(SPACE_SAVE_DATA_ROOT, "all-new-pkls", "sr-pkl")
    
    all_sents = pickle.load(open(os.path.join(sum_dir, "all-new.pkl"), 'rb'))
    stemmer = nltk.stem.PorterStemmer()
    
    words = set()
    counts = {}
    n = 0
    for sent in all_sents:
        n += 1
        where = set()
        for word in word_tokenize(sent):
            if word in stopwords.words():
                continue
            word = stemmer.stem(word)
            if word not in words:
                words.add(word)
                counts[word] = 0
            if word not in where:
                where.add(word)
                counts[word] += 1
        
        if n % 100 == 0:
            print("{}/{} done.".format(n, len(all_sents)))
    
    for word in counts:
        counts[word] = math.log(n/(counts[word]+1))
    
    for name in pkl_names:
        in_path = os.path.join(sum_dir, name)
        summarized = pickle.load(open(in_path, 'rb'))
        avg = 0
        n = 0
        for eid in summarized:
            for aspect in summarized[eid]:
                for sentence in sent_tokenize(summarized[eid][aspect]):
                    words = [stemmer.stem(word) for word in word_tokenize(sentence) if \
                        word not in stopwords.words()]
                    scores = [counts[word] for word in words]
                    avg += sum(scores, 0)/len(scores)
                n += 1
        print("{}  :  {}".format(name[:-4], (avg/n)))    