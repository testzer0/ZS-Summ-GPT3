from globals import *
from entailment import format_summaries
from sentiment import get_positive_negative_nums
from data import read_space_data

import nltk
from nltk.tokenize import sent_tokenize
import json
import random
import numpy as np
import os
import sys
import pickle

from rouge_score import rouge_scorer
from bert_score import score

_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def get_scores(reference, output):
    """
    Returns Rouge-1 F1, Rouge-L F1 and BERT Scores 
    """
    # rscores = _scorer.score(reference, output)
    # R1 = rscores['rouge1'][2]
    # RL = rscores['rougeL'][2]
    _, _, F1 = score([output], [reference], lang='en', verbose=False)
    return F1[0]

def save_outputs_with_goldens(entities, summaries, sentiments, fname, multiple_levels=False,
    dir="summaries"):
    cluster_file = os.path.join(SPACE_SAVE_DATA_ROOT, fname)
    tas = pickle.load(open(cluster_file, 'rb'))
    root = os.path.join(SPACE_SAVE_DATA_ROOT, dir)
    if not os.path.exists(root):
        os.mkdir(root)
    for entity in entities:
        eid = entity['entity_id']
        entity_dir = os.path.join(root, str(eid))
        if not os.path.exists(entity_dir):
            os.mkdir(entity_dir)
        aspect_mapping = {aspect: [] for aspect in aspects_+["none"]}
        sentences = tas[eid]
        for (topics, aspects, sentence) in sentences:
            for aspect in aspects:
                aspect_mapping[aspect].append(sentence)
        for aspect, summary in summaries[eid]:
            if aspect != "none":
                sentences = ["Aspect: {}".format(aspect)]
                sentences.append("***** Sentiments *****")
                pos, neg, tot = sentiments[eid][aspect]
                sentences.append("Statements had {} positive and {} negative ({} total)".\
                    format(pos, neg, tot))
            if multiple_levels and len(summary) > 1:
                if aspect != "none":
                    s_pos_c, s_neg_c, _ = get_positive_negative_nums(sent_tokenize(\
                        " ".join(summary[0])), aspect, combine=True)
                    sentences.append("Summaries when concatenated have {} positive and {} negative".\
                        format(s_pos_c, s_neg_c))
                    s_pos_s, s_neg_s, _ = get_positive_negative_nums(sent_tokenize(summary[-1]), \
                        aspect, combine=True)
                    sentences.append("Summaries when summarized again have {} positive and {} negative".\
                        format(s_pos_s, s_neg_s))
                sentences.append("***** Model produced summary in the 1st round *****")
                for i in range(len(summary[0])):
                    sentences.append("Part {} of {}: {}".format(i+1, \
                        len(summary[0]), summary[0][i]))
                sentences.append("***** Model produced summary - final *****")
                summary = summary[-1]
            else:
                if multiple_levels:
                    summary = summary[-1]
                if aspect != "none":
                    s_pos_s, s_neg_s, _ = get_positive_negative_nums(sent_tokenize(summary), \
                        aspect, combine=True)
                    sentences.append("Summary has {} positive and {} negative".\
                        format(s_pos_s, s_neg_s))
                sentences.append("***** Model produced summary *****")
            sentences.append(summary)
            if aspect != "none":
                goldens = entity['summaries'][aspect]
                i = 1
                for golden in goldens:
                    sentences.append("***** Golden Review {} *****".format(i))
                    i += 1
                    sentences.append(golden)
            sentences.append("***** Sentences in Cluster ****")
            sentences += aspect_mapping[aspect]
            with open(os.path.join(entity_dir, "{}.txt".format(aspect)), "w+") as f:
                f.write("\n".join(sentences))          
    
def calculate_scores_wrt_goldens(entities, summaries):
    _aspects = aspects_[1:]
    BSs = []
    aspect_average_BS = {aspect: 0 for aspect in _aspects}
    average_BS = 0
    n_entities = 0
    for entity in entities:
        if entity['entity_id'] not in summaries:
            continue
        n_entities += 1
        s = summaries[entity['entity_id']]
        BS = []
        for aspect in s:
            summary = s[aspect]
            goldens = entity['summaries'][aspect]
            best_BS = 0
            for golden in goldens:
                BSv = get_scores(golden, summary)
                best_BS = max(best_BS, BSv)
            aspect_average_BS[aspect] += best_BS
            BS.append(best_BS)
        average_BS += sum(BS) / len(BS)
        BSs.append(BS)
    average_BS /= n_entities
    for aspect in _aspects:
        aspect_average_BS[aspect] /= n_entities
        print("Aspect - {}".format(aspect))
        print("Average BERT score F1 is {}".format(aspect_average_BS[aspect]))
        print("**********")
    print("Overall, Average BERT score F1 is {}".format(average_BS))
    return BSs
    
if __name__ == '__main__':
    pkl_names = ['tcg.pkl', 'tqg.pkl', 'qg.pkl', 'acesum.pkl', 'qfsumm.pkl', 'rg.pkl']
    sum_dir = os.path.join(SPACE_SAVE_DATA_ROOT, "all-new-pkls", "summaries-pkl")
    for name in pkl_names:
        in_path = os.path.join(sum_dir, name)
        entities = read_space_data()
        # save_outputs_with_goldens(entities, summaries, sentiments, "topic-annotated.pkl", 
        #    multiple_levels=False, dir="qfsumm-long-gpt3-summarized")
        _ = calculate_scores_wrt_goldens(entities, summaries)