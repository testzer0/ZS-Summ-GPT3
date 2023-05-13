"""
Functions related to aspect-specific clustering of sentences.
"""
from re import A
import ssl
from urllib import response
from matplotlib import use
from globals import *
from data import get_aspect, read_space_data, get_prompt, \
    map_keyword_to_closest_aspect, get_gpt3_response
import numpy as np
import random
import os
import pickle
import math

from sklearn.cluster import KMeans

from kneed import KneeLocator

import nltk
import json
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_glove_word(word):
    if word in GLOVE:
        return word
    elif " " in word:
        return get_glove_word(word.split(" ")[0].strip())
    for i in range(len(word), 3, -1):
        if word[:i] in GLOVE:
            return word[:i]
    return "none"

def best_kmeans_for_words(words_or_vectors, kmin=3, kmax=12):
    """
    First maps each word to its GloVe vector. Then for each k from kmin to kmax
    (inclusive) KMeans clustering is done and the calinski harabasz score evaluated.
    The knee point of the formed curve is found using kneedle, and that value of k
    is chosen as optimal. The corresponding k, and labels are returned.
    """
    vectors = np.concatenate([GLOVE[word] for word in words], axis=0)
    labels = []
    inertias = []
    ks = []
    for k in range(kmin, kmax+1):
        ks.append(k)
        kmeans = KMeans(n_clusters=k, random_state=2003).fit(vectors)
        labels.append(kmeans.labels_)
        inertias.append(kmeans.inertias_)
    kneedle = KneeLocator(ks, inertias, curve="convex", direction="decreasing")    
    k = int(kneedle.elbow)
    return k, labels[k-kmin]

def cluster_based_on_review_score(entities):
    clusters = {}
    for entity in entities:
        eid = entity['entity_id']
        sentences = {}
        for review in entity['reviews']:
            rating = review['rating']
            if rating not in sentences:
                sentences[rating] = rating['sentences']
            else:
                sentences[rating] += rating['sentences']
        clusters[eid] = sentences
    return clusters

def get_topics_for_sentence(sentence):
    prompt_part = get_prompt("hotel-topic")
    prompt = prompt_part + "\nSentence: {}\nTopic:".format(sentence)
    topics = get_gpt3_response(prompt).lower()
    while " and " in topics:
        i = topics.find(" and ")
        topics = topics[:i] + "," + topics[i+5:]
    if "," in topics:
        topics = [topic.strip() for topic in topics.split(",")]
    else:
        topics = [topics]
    topics = [get_glove_word(topic) for topic in topics]
    return topics    

def get_topic_annotated_sentences_for_entities(entities):
    topic_annotated_sentences = {}
    for entity in entities:
        entity_id = entity['entity_id']
        sentences = []
        for review in entity['reviews']:
            for sentence in review['sentences']:
                topics = get_topics_for_sentence(sentence)
                aspects = [map_keyword_to_closest_aspect(topic) \
                    for topic in topics]
                sentences.append((topics, aspects, sentence))
        topic_annotated_sentences[entity_id] = sentences
    return topic_annotated_sentences

def get_gpt3_summary(aspect, sentences, group_size=30, is_summarized=False, \
    return_all_levels=False):
    if len(sentences) > group_size:
        n_groups = (len(sentences)+group_size-1) // group_size
        group_size = len(sentences) // n_groups
        summaries = []
        ss = []
        cur = 0
        while cur < len(sentences):
            if cur + group_size >= len(sentences):
                summary = get_gpt3_summary(aspect, \
                    sentences[cur:], group_size, is_summarized)
            else:
                summary = get_gpt3_summary(aspect, \
                    sentences[cur:cur+group_size], group_size, is_summarized)
            summaries.append(summary)
            ss += sent_tokenize(summary)
            cur += group_size
        if return_all_levels:
            return [summaries] + get_gpt3_summary(aspect, ss, \
                group_size, is_summarized=True, return_all_levels=True)
        else:
            return get_gpt3_summary(aspect, ss, group_size, is_summarized=True)
    if is_summarized:
        prompt = "Here are some accounts of the reviews of a hotel:\n\n"
    else:
        prompt = "Here are some reviews of a hotel:\n\n"
    joined = "\n".join(sentences)
    if len(joined) > 3400:
        # Truncate if the length is too much
        joined = joined[:3400]
        joined = joined[:joined.rfind('\n')]
    prompt += joined+"\n\n"
    if is_summarized:
        prompt += "Summarize what the accounts said of the {}:".format(aspect)
    else:
        prompt += "Summarize what the reviews said of the {}:".format(aspect)
    if return_all_levels:
        return [[get_gpt3_response(prompt)]]
    else:
        return get_gpt3_response(prompt)

def get_gpt3_summaries(clusters, useall=False):
    summaries = {}
    for eid in clusters:
        if useall:
            sentences = []
        else:
            sentences = {aspect: [] for aspect in aspects_[1:]}
        summary = {}
        for (topics, aspects, s) in clusters[eid]:
            if useall:
                sentences.append(s)
            else:
                for aspect in aspects:
                    if aspect != "none":
                        sentences[aspect].append(s)
        for aspect in aspects_[1:]:
            summary[aspect] = get_gpt3_summary(aspect, sentences if useall \
                else sentences[aspect], is_summarized=False, return_all_levels=True)
        summaries[eid] = summary
    return summaries

def get_aspectwise_review_stratified(entities, ac):
    mapping = {}
    for entity in entities:
        eid = entity['entity_id']
        if eid not in ac:
            continue
        rating_map = {}
        ratings = set()
        for review in entity['reviews']:
            rating = review['rating']
            ratings.add(rating)
            for sentence in review['sentences']:
                rating_map[sentence] = rating
        entity_sentences = []
        for aspect in aspects_[1:]:
            aspect_sentences = ac[eid][aspect]
            clusters = []
            for rating in ratings:
                sentences = [sentence for sentence in aspect_sentences if \
                    rating_map[sentence] == rating]
                if len(sentences) > 0:
                    clusters.append(sentences)
            entity_sentences.append((aspect, clusters))
        mapping[eid] = entity_sentences
    return mapping

def get_aspectwise_review_stratified_gpt3_summaries(mapping):
    summaries = {}
    for eid in mapping:
        summary = {}
        for (aspect, clusters) in mapping[eid]:
            aspect_summaries = []
            for i, cluster in enumerate(clusters):
                name = "temp/old7/{}-{}-{}.pkl".format(eid, aspect, i)
                if os.path.exists(name):
                    print("Found "+name+".")
                    aspect_summaries.append(pickle.load(open(name, 'rb')))
                else:
                    # Pass large group size to have it all in one group
                    s = get_gpt3_summary(aspect, cluster, \
                        group_size=10000, is_summarized=False, return_all_levels=False)
                    aspect_summaries.append(s)
                    pickle.dump(s, open(name, 'wb+'))
                    print(name+" done.")
            # Also get the combined summary
            name = "temp/old7/{}-{}-comb.pkl".format(eid, aspect)
            if os.path.exists(name):
                print("Found "+name+".")
                combined = pickle.load(open(name, 'rb'))
            else:
                combined = get_gpt3_summary(aspect, aspect_summaries, group_size=10000, \
                    is_summarized=True, return_all_levels=False)
                pickle.dump(combined, open(name, 'wb+'))
                print(name+" done.")
            summary[aspect] = [aspect_summaries, combined]
            print("Aspect {} for {} done.".format(aspect, eid))
            print(combined+"\n\n----------------\n")
        summaries[eid] = summary
        print("EID {} done.".format(eid))
    return summaries
                

def do_sr(sentence):
    sr_prompt = "Split and Rephrase the following sentences into simple value judgements. "+\
        "Use multiple statements instead of using connectives, and avoid references to people "+\
        "making the judgement. Prefer specific features to a general judgement as good "+\
        "or bad.\n\nSentence: The hotel is generally well received, "+\
        "though some found issues with bland or cold food and stained carpets.\n"+\
        "Output:\nThe hotel is good.\nThe food is bland.\nThe food is cold.\nThe "+\
        "carpets have stains.\n\nSentence: {}\nOutput:".format(sentence)
    response = get_gpt3_response(sr_prompt).strip()
    return [sent.strip() for sent in sent_tokenize(response)]

def test():
    pkl_names = ['tcg.pkl', 'tqg.pkl', 'qg.pkl', 'acesum.pkl', 'qfsumm.pkl']
    pkl_names = ['tqg.pkl', 'qg.pkl', 'qfsumm.pkl']
    sum_dir = os.path.join(SPACE_SAVE_DATA_ROOT, "all-new-pkls", "summaries-pkl")
    sr_dir = os.path.join(SPACE_SAVE_DATA_ROOT, "all-new-pkls", "sr-pkl")
    
    for name in pkl_names:
        in_path = os.path.join(sum_dir, name)
        out_path = os.path.join(sr_dir, name)
        print("Starting {}.\n".format(name[:-4]))
        sums = pickle.load(open(in_path, 'rb'))
        sr = {}
        for eid in sums:
            sr_eid = {}
            for aspect in sums[eid]:
                # If we crashed last time, pick up where we left off
                if os.path.exists("temp/{}-{}.pkl".format(eid, aspect)):
                    print("Found {}-{}.".format(eid, aspect))
                    sr_eid[aspect] = pickle.load(open("temp/old7/{}-{}.pkl".format(eid, \
                        aspect), 'rb'))
                    continue
                else:
                    print("Start {}-{}.\n".format(eid, aspect))
                sr_sent = []
                for sent in sent_tokenize(sums[eid][aspect]):
                    parts = do_sr(sent)
                    sr_sent.append((sent, parts))
                    print("{} was split into:".format(sent))
                    for part in parts:
                        print(part)
                    print()
                sr_eid[aspect] = sr_sent
                pickle.dump(sr_sent, open("temp/{}-{}.pkl".format(eid, aspect), 'wb+'))
                print("Finish {}-{}.\n".format(eid, aspect))
            sr[eid] = sr_eid
        pickle.dump(sr, open(out_path, 'wb+'))    
        print("Done {}.\n".format(name[:-4]))
        os.system("rm temp/*")      
    
if __name__ == '__main__':
    test()