"""
Handles loading of data, its (pre-)processing and also the interfacing with GPT-3.
"""

from globals import *
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import random
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# from keybert import KeyBERT

aspect_vecs_ = [GLOVE[aspect] for aspect in aspects_+["none"]]
        
def read_space_data(file_name="space_summ.json", as_classes=False):
    file_path = os.path.join(SPACE_DATSET_ROOT, file_name)
    json_data = json.load(open(file_path))
    if not as_classes:
        return json_data
    entities = []
    for entity in json_data:
        reviews = []
        for review in entity['reviews']:
            reviews.append(Review(review['sentences'], review['review_id'], review['rating']))
        entities.append(Entity(reviews, entity['summaries'], entity['entity_id'], \
            entity['entity_name']))
    return entities

def print_statistics(file_name="space_summ.json"):
    entities = read_space_data()
    n_entities = len(entities)
    n_reviews = 0
    n_sentences = 0
    for entity in entities:
        n_reviews += len(entity['reviews'])
        for review in entity['reviews']:
            n_sentences += len(review['sentences'])
    print("{} entities, {} reviews and {} sentences.".format(n_entities, \
        n_reviews, n_sentences))
    print("On average,")
    print("   {:.2f} sentences per review".format(n_sentences / n_reviews))
    print("   {:.2f} reviews per entity".format(n_reviews / n_entities))
    print("   {:.2f} sentences per entity".format(n_sentences / n_entities))
    
def get_prompt(name):
    with open(os.path.join(PROMPT_DIR, name+".txt")) as f:
        return "\n".join([line for line in f])

def get_gpt3_response(prompt, tokenize=False):
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, \
                                  temperature=0.7, max_tokens=256) 
    response = response.choices[0].text.strip()
    if tokenize:
        response = sent_tokenize(response)
    return response          

def map_keyword_to_closest_aspect(keyword):
    if keyword not in GLOVE:
        return "none"
    kvec = GLOVE[keyword]
    distances = [np.linalg.norm(aspect_vec - kvec) for \
        aspect_vec in aspect_vecs_[1:]]
    chosen = distances.index(min(distances))
    return "none" if chosen == len(distances)-1 else aspects_[chosen+1]

def cosine_similarity(w1, w2):
    if w1 == w2:
        return 1
    elif w1 not in GLOVE or w2 not in GLOVE:
        return -1
    v1 = GLOVE[w1] 
    v1 = v1 / np.linalg.norm(v1)
    v2 = GLOVE[w2]
    v2 = v2 / np.linalg.norm(v2)
    return np.dot(v1, v2)

def sentence_has_keyword(sentence, keyword, exact=True, threshold=0.75):
    sentence = sentence.lower()
    keyword = keyword.lower()
    if exact:
        """
        Directly check substring existence, since 'room' might become 'rooms' and so on.
        """
        return (keyword in sentence)
    else:
        words = [word for word in sentence.split(" ") if \
            cosine_similarity(word, keyword) > threshold]
        return not words.empty()

def get_aspect(topic):
    topic = topic.lower()
    # Handle some special cases first
    for w in ['pool', 'lobby', 'building', 'lounge', 'parking']:
        if w in topic:
            return 'building'
    for w in ['clean', 'dirty', 'stain', 'dust', 'tidy']:
        if w in topic:
            return 'cleanliness'
    for w in ['room', 'bath', 'bed', 'heater', 'blanket']:
        if w in topic:
            return 'rooms'
    for w in ['service', 'staff', 'manager', 'reception']:
        if w in topic:
            return 'service'
    for w in ['location', 'view', 'near', 'near', 'close']:
        if w in topic:
            return 'location'
    for w in ['food', 'breakfast', 'lunch', 'dinner', 'snacks', 'restaurant', \
        'meal', 'serve']:
        if w in topic:
            return 'food'
    if ' ' in topic:
        topic = topic[:topic.index(' ')]
    return map_keyword_to_closest_aspect(topic)

def prompt_for_keywords(text):
    print("Summary:")
    print(text)
    print("Enter keywords: ", end='')
    keywords = input()
    keywords = [keyword.strip() for keyword in keywords.strip().lower().split(",")]
    keywords = [keyword for keyword in keywords if keyword != '']
    return keywords

def get_ks():
    gpt3_summaries = pickle.load(open(os.path.join(SPACE_SAVE_DATA_ROOT, \
        "gpt3-summarized-alllevels.pkl"), 'rb'))
    keywords = {}
    for eid in gpt3_summaries:
        keywords_for_eid_concat = {}
        keywords_for_eid_summarized = {}
        for aspect in gpt3_summaries[eid]:
            summary = gpt3_summaries[eid][aspect]
            keywords_for_eid_summarized[aspect] = prompt_for_keywords(summary[-1])
            if len(summary) > 1:
                keywords_for_eid_concat[aspect] = \
                    prompt_for_keywords(" ".join(summary[0]))
            else:
                keywords_for_eid_concat[aspect] = keywords_for_eid_summarized[aspect]
        keywords[eid] = (keywords_for_eid_concat, keywords_for_eid_summarized)
    pickle.dump(keywords, open(os.path.join(SPACE_SAVE_DATA_ROOT, "keywords-gpt3-all.pkl"), 'wb+'))

def test():
    x = open(os.path.join(SPACE_SAVE_DATA_ROOT, 'topic-annotated.pkl'), 'rb')
    y = pickle.load(x)
    n = 0
    mn, mx, avg = 1e9, 0, 0
    for eid in y:
        mapping = {a: [] for a in aspects_[1:]}
        for (_,aspects,s) in y[eid]:
            for a in aspects:
                if a != 'none':
                    mapping[a].append(s)
        for a in mapping:
            l = len(mapping[a])
            mx = max(mx, l)
            mn = min(mn, l)
            avg += l
            n += 1
    print(mn, mx, avg*1.0/n)

if __name__ == '__main__':
    summarized = pickle.load(open(os.path.join(SPACE_SAVE_DATA_ROOT, \
        "summaries-pkl/qfsumm-summarized.pkl"), 'rb'))
    keys = list(summarized.keys())
    s = read_space_data()
    space = {}
    reviews = {}
    for entry in s:
        if entry['entity_id'] in keys:
            space[entry['entity_id']] = entry
    combos = [('185804', 'rooms'), ('100597', 'cleanliness'), ('183092', 'location'), \
        ('183092', 'service'), ('1113787', 'rooms'), ('120274', 'cleanliness'), \
        ('1176198', 'building'), ('112429', 'food'), ('1029276', 'location'), \
        ('100597', 'service')]
    for eid, aspect in combos:
        print(eid, aspect)
        # print(summarized[eid][aspect][-1])
        for a, summary in summarized[eid]:
            if a != aspect:
                continue 
            print(summary)
        print("------------")