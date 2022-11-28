from globals import *
from credentials import *

import os
import pickle
import nltk
import math
import json
import csv
import numpy as np

from nltk import sent_tokenize, word_tokenize

from bert_score import score

def get_scores(reference, output):
    """
    Returns Rouge-1 F1, Rouge-L F1 and BERT Scores 
    """
    # rscores = _scorer.score(reference, output)
    # R1 = rscores['rouge1'][2]
    # RL = rscores['rougeL'][2]
    _, _, F1 = score([output], [reference], lang='en', verbose=False)
    return F1[0]

def get_gpt3_response(prompt, tokenize = False):
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, \
                                  temperature=0.7, max_tokens=256) 
    response = response.choices[0].text.strip()
    if tokenize:
        response = sent_tokenize(response)
    return response       

def parse_dataset(dataset_name="amazon"):
    """
    Parse the various csv files into a common dataset.
    """
    dataset_name = dataset_name.lower()
    assert(dataset_name in ["amazon", "yelp"])
    dataset_root = os.path.join(FS_DATASET_ROOT, "artifacts/{}/reviews".format(dataset_name))
    dataset = {}
    for split_name in ['train', 'val']:
        split = {}
        split_root = os.path.join(dataset_root, split_name)
        for fname in os.listdir(split_root):
            entity_id = None
            with open(os.path.join(split_root, fname)) as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = []
                for row in reader:
                    if entity_id is None:
                        entity_id = row['group_id']
                        entity_id = entity_id[:entity_id.rfind('_')]
                    row['group_id'] = int(row['group_id'][row['group_id'].rfind('_')+1:])
                    rows.append(row)
                if entity_id in split:
                    split[entity_id] += rows 
                else:
                    split[entity_id] = rows                      
        dataset[split_name] = split
    return dataset

def load_dataset(dataset_name, original=False):
    """
    Loads the pickled dataset.
    """
    assert(dataset_name in ["amazon", "yelp"])
    if not original:
        return pickle.load(open(os.path.join(FS_SAVE_DATA_ROOT, \
            dataset_name+"-subset.pkl"), 'rb'))
    else:
        return pickle.load(open(os.path.join(FS_SAVE_DATA_ROOT, dataset_name+".pkl"), 'rb'))

def print_statistics(dataset):
    """
    Prints basic statistics about the dataset
    """
    for split in dataset:
        print("Split {}:".format(split))
        n_entities = len(dataset[split].keys())
        n_reviews = 0
        n_sentences = 0
        n_words = 0
        for eid in dataset[split]:
            n_reviews += len(dataset[split][eid])
            for review in dataset[split][eid]:
                sents = sent_tokenize(review['review_text'])
                n_sentences += len(sents)
                for sent in sents:
                    n_words += len(word_tokenize(sent))
        print("Number of entities = {}".format(n_entities))
        print("Number of reviews = {} (on average {} reviews/entity".format(n_reviews, \
            n_reviews/n_entities))
        print("Number of sentences = {} (on average {} sentences/review)".format(n_sentences, \
            n_sentences/n_reviews))
        print("Number of words = {} (on average {} words/review)".format(n_words, \
            n_words/n_reviews))        

def gpt3_summarize_single_product(reviews, dataset_name, already_summarized=False):
    dataset_name = dataset_name[0].upper() + dataset_name[1:]
    if already_summarized:
        prompt = "Here are some accounts of the reviews of a product at {}:\n\n".\
            format(dataset_name)
    else:
        prompt = "Here are some reviews of a product at {}:\n\n".format(dataset_name)
    for review in reviews:
        if type(review) == str:
            prompt += review+'\n'
        else:
            prompt += review['review_text']+'\n'
        if len(word_tokenize(prompt)) >= 3200:
            break
    if already_summarized:
        prompt += "\nSummarize what the accounts said of the product:"
    else:
        prompt += "\nSummarize what the reviews said of the product:"
    return get_gpt3_response(prompt)

def gpt3_keywords_single_product(reviews, dataset_name):
    dataset_name = dataset_name[0].upper() + dataset_name[1:]
    prompt = "Here are some reviews of a product at {}:\n\n".format(dataset_name)
    for review in reviews:
        if type(review) == str:
            prompt += review + '\n'
        else:
            prompt += review['review_text']+'\n'
        if len(word_tokenize(prompt)) >= 3200:
            break
    prompt += "\nOutput comma-separated keywords that capture the reviews most comprehensively:"
    response = get_gpt3_response(prompt)
    return [keyword.strip() for keyword in response.strip().split(',')]

def gpt3_split_single_sentence(sentence):
    prompt = "Split the following sentences into simple propositions.\n" + \
        "Sentence:\n" + \
        "However, some people found the fabric to be cheap-looking or unflattering, " + \
            "and there were a few complaints about the pleats not being well-done.\n" + \
        "Propositions:\n" + \
        "Some people found the fabric to be cheap looking.\n" + \
        "Some people found the fabric to be unflattering.\n" + \
        "There were a few complaints about the pleats not being well-done.\n" + \
        "Sentence:\n" + \
        "There were a few complaints about the product being damaged or not " + \
            "as described, but generally speaking, " + \
            "reviewers were happy with the product.\n" + \
        "Propositions:\n" + \
        "There were a few complaints about the product being damaged.\n" + \
        "There were a few complaints about the product being not as described.\n" + \
        "Reviewers were happy with the product.\n" + \
        "Sentence:\n{}\nPropositions:".format(sentence)
    response = get_gpt3_response(prompt)
    return [sentence.strip() for sentence in sent_tokenize(response)]

def gpt3_rephrase_single_sentence(sentence):
    prompt = "Rephrase the following sentences as simple value judgements.\n" + \
        "Sentence: Some people found it to be a great gift for lawyers.\n" + \
        "Output: It was a great gift for lawyers.\n" + \
        "Sentence: Many people found that the pill helped " + \
            "lower their cholesterol levels.\n" + \
        "Output: The pill helped lower cholesterol levels.\n" + \
        "Sentence: {}\nOutput:".format(sentence)
    response = get_gpt3_response(prompt)
    return [sentence.strip() for sentence in sent_tokenize(response)]

def gpt3_summarize_multilevel_single_product(reviews, dataset_name, \
    already_summarized=False, k=35, return_all_levels=False, \
    different_format=False):
    if already_summarized or different_format:
        sentences = reviews
    else:
        sentences = []
        for review in reviews:
            sentences += sent_tokenize(review['review_text'])
    if len(sentences) <= k:
        return gpt3_summarize_single_product(sentences, dataset_name, \
            already_summarized=already_summarized)
    n_blocks = math.ceil(len(sentences) / 30)
    smaller_block_size = math.floor(len(sentences)/n_blocks)
    n_larger_blocks = len(sentences) % smaller_block_size
    total_larger = n_larger_blocks*(smaller_block_size+1)
    summary_sentences = []
    for i in range(0, total_larger, smaller_block_size+1):
        summary = gpt3_summarize_single_product(sentences[i:i+smaller_block_size+1], \
            dataset_name, already_summarized=already_summarized)
        summary_sentences += sent_tokenize(summary)
    for i in range(total_larger, len(sentences), smaller_block_size):
        summary = gpt3_summarize_single_product(sentences[i:i+smaller_block_size], \
            dataset_name, already_summarized=already_summarized)
        summary_sentences += sent_tokenize(summary)
    subresult = gpt3_summarize_multilevel_single_product(summary_sentences, dataset_name, \
        already_summarized=True, k=k, return_all_levels=return_all_levels)
    ###
    # If return_all_levels is True, the resultant will be of the form
    # [[l1_sent1, ..., l1_sentk], ..., [l(last-1)_sent1, ... l(last-1)_sentm], last_sum]
    # All except the last entry are lists of sentences; the last is a string of the entire
    # summary
    ###
    if return_all_levels:
        if type(subresult) == str:
            # Was the final level
            return [summary_sentences, subresult]
        else:
            # The list form has already been initiated
            return [summary_sentences] + subresult
    else:
        return subresult

def gpt3_summarize(dataset_name, dataset=None):
    if dataset is None:
        dataset = load_dataset(dataset_name)
    summaries = {}
    for product in dataset:
        summaries[product] = gpt3_summarize_single_product(dataset[product], dataset_name)
    return summaries

def gpt3_summarize_multilevel(dataset_name, dataset=None, return_all_levels=False):
    if dataset is None:
        dataset = load_dataset(dataset_name)
    summaries = {}
    for product in dataset:
        summaries[product] = gpt3_summarize_multilevel_single_product(dataset[product], \
            dataset_name, return_all_levels=return_all_levels)
    return summaries

def gpt3_get_keywords(dataset_name, dataset=None):
    if dataset is None:
        dataset = load_dataset(dataset_name)
    keywords = {}
    for product in dataset:
        keywords[product] = gpt3_keywords_single_product(dataset[product], dataset_name)
    return keywords

def filter_dataset(dataset, split='test'):
    return {
        key : dataset[key] for key in dataset if dataset[key]['split'] == split
    }

if __name__ == '__main__':
    ent_dir = os.path.join(FS_SAVE_DATA_ROOT, "entailment-pkl")
    sr_dir = os.path.join(FS_SAVE_DATA_ROOT, "rephrased-pkl")
    sum_dir = os.path.join(FS_SAVE_DATA_ROOT, "summaries-pkl")
    summarized = pickle.load(open(os.path.join(sum_dir, 'fewsum-yelp.pkl'), 'rb'))
    out_path = os.path.join(sr_dir, 'fewsum-yelp.pkl')
    sr_out = {}
    for eid in summarized:
        summary = summarized[eid]
        cur_sr_out = []
        for sent in sent_tokenize(summary):
            sent_out = []
            parts = gpt3_split_single_sentence(sent)
            for part in parts:
                sent_out += gpt3_rephrase_single_sentence(parts)
            cur_sr_out.append((sent, sent_out))
        sr_out[eid] = cur_sr_out  
    pickle.dump(sr_out, open(out_path, 'wb+'))