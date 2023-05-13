from globals import *
from credentials import *
from data import load_dataset

import os
import pickle
import nltk
import math
import json
import csv
import numpy as np

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from bert_score import score

try:
    from model_summac import SummaCConv, SummaCZS
    from simcse import SimCSE
except ImportError:
    pass

def get_scores(reference, output):
    """
    Returns Rouge-1 F1, Rouge-L F1 and BERT Scores 
    """
    # rscores = _scorer.score(reference, output)
    # R1 = rscores['rouge1'][2]
    # RL = rscores['rougeL'][2]
    _, _, F1 = score([output], [reference], lang='en', verbose=False)
    return F1[0]

model = None
def get_model(type="conv"):
    if type.lower() == "conv":
        return SummaCConv(granularity="sentence")
    elif type.lower() == "simcse":
        return SimCSE("princeton-nlp/sup-simcse-roberta-large")
    else:
        return SummaCZS(granularity="sentence", model_name="vitc")
    
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
    
def get_top_supporting_and_weakening_lines(sentences, summary, k=5, model_type="zs", \
    filter_len=False):
    global model
    if model is None:
        model = get_model(model_type)
    if type(summary) == str:
        summary = sent_tokenize(summary)
        if filter_len:
            summary = [line.strip() for line in summary if len(line.strip().split(" ")) >= 4]
        else:
            summary = [line.strip() for line in summary]
    result = []
    linewise_scores = []        # This is used only in the case of simcse
    for line in summary:
        if model_type.lower() == "simcse":
            sentences_with_scores = [(sentence, model.similarity([sentence], [line])[0][0]) \
                for sentence in sentences]
        else:
            sentences_with_scores = [(sentence, model.score([sentence], [line])["scores"][0]) \
                for sentence in sentences]
        sentences_with_scores = sorted(sentences_with_scores, key=lambda x : x[1])
        worst_k = sentences_with_scores[:k]
        best_k = list(reversed(sentences_with_scores[-k:]))
        result.append((line, best_k, worst_k))
        linewise_scores.append(best_k[0][1])
    overall_score = 0 if len(linewise_scores) == 0 else \
        sum(linewise_scores)*1.0/len(linewise_scores)
    return result, linewise_scores, overall_score

def is_simple_sentence(sentence):
    words = word_tokenize(sentence)
    for word in ['while', 'but', 'though', 'although', 'other', 'others', \
        'however']:
        if word in words:
            return False
    return True

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

def average_abstractiveness(dataset, summaries, ns=[3,4,5]):
    avg_abstractiveness = [0]*len(ns)
    for product in summaries:
        review_sentences = []
        for review in dataset[product]:
            review_sentences += sent_tokenize(review['review_text'])
        summary_sentences = sent_tokenize(summaries[product])
        for i in range(len(ns)):
            avg_abstractiveness[i] += abstractiveness(summary_sentences=summary_sentences, \
                reference_sentences=review_sentences, n=ns[i])
    for i in range(len(avg_abstractiveness)):
        avg_abstractiveness[i] /= len(list(summaries.keys()))
    return avg_abstractiveness

def get_num_simple_and_complex_sentences(summaries):
    n_simple, n_complex = 0, 0
    for product in summaries:
        summary = summaries[product]
        if type(summary) == str:
            summary = sent_tokenize(summary)
        for sentence in summary:
            if is_simple_sentence(sentence):
                n_simple += 1
            else:
                n_complex += 1
    return n_simple, n_complex

def save_and_return_entailment_scores(dataset, summaries, root_dir, model_type='zs', \
    split_or_rephrased=True, k=5, orig_summaries=None):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    if orig_summaries is None:
        orig_summaries = summaries
    lines_with_scores = {}
    for product in summaries:
        summary = summaries[product]
        parents = []
        if split_or_rephrased:
            temp = []
            for (sentence, parts) in summary:
                temp += parts
                parents += [sentence] * len(parts)
            summary = temp
        else:
            if type(summary) == str:
                summary = sent_tokenize(summary)
            parents = summary
        review_sentences = []
        output = ["Summary:", "", orig_summaries[product], ""]
        for review in dataset[product]:
            review_sentences += sent_tokenize(review['review_text'])
        lines, scores, total = get_top_supporting_and_weakening_lines(review_sentences, \
            summary, k=k, model_type=model_type)
        lines_with_scores[product] = (lines, parents, scores, total)
        for (line, supporting, weakening), parent, score in zip(lines, parents, scores):
            print("          Line {}".format(line))
            output.append("**********")
            output.append("[{:.4f}] {}".format(score, line))
            output.append("Parent: {}".format(parent))
            output.append("")
            output.append("Top {} (supporting):".format(k))
            for sentence, supp_score in supporting:
                output.append("     [{:.4f}] {}".format(supp_score, sentence))
            output.append("")
            output.append("Top {} (weakening):".format(k))
            for sentence, weak_score in weakening:
                output.append("     [{:.4f}] {}".format(weak_score, sentence))
            output.append("")
        with open(os.path.join(root_dir, product), 'w+') as f:
            f.write("\n".join(output))
        print("    Product {} done.".format(product))
    return lines_with_scores

def average_top_score(lines_with_scores):
    avg_score = 0
    n_lines = 0
    for product in lines_with_scores:
        n_lines += len(lines_with_scores[product][2])
        avg_score += sum(lines_with_scores[product][2])
    return avg_score/n_lines

def support_sizes(lines_with_scores, thresh=0.5):
    counts = [0]*6
    for product in lines_with_scores:
        for (_, supporting, _) in lines_with_scores[product][0]:
            cnt = sum([1 if elem[1] >= thresh else 0 for elem in supporting])
            counts[cnt] += 1
    total = sum(counts)
    for i in range(6):
        counts[i] *= 100.0/total
    return counts

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
    entailment_dir = os.path.join(FS_SAVE_DATA_ROOT, "entailment-pkl")
    summaries_dir = os.path.join(FS_SAVE_DATA_ROOT, "summaries-pkl")
    split_dir = os.path.join(FS_SAVE_DATA_ROOT, "split-pkl")
    rephrased_dir = os.path.join(FS_SAVE_DATA_ROOT, "rephrased-pkl")
    in_path = os.path.join(entailment_dir, 'fewsum-yelp-new.pkl')
    dataset = load_dataset('yelp')
    summaries = json.load(open('../../datasets/FewSum/temp/sums-yelp-tv.json'))
    
    ref_root = '../rouge/v1.2.2/projects/test-summarization/reference'
    ref_files = []
    for i in range(1,4):
        ref_files.append(os.path.join(ref_root, 'task1_ref{}.txt'.format(i)))
    sum_file = '../rouge/v1.2.2/projects/test-summarization/system/task1_sys.txt'
    command = 'java -jar ../rouge/v1.2.2/rouge2-1.2.2.jar > /dev/null'
    csv_file = '../rouge/v1.2.2/results.csv'                

    rouge_for_name = []
    for split in  ['val', 'test']:
        f = json.load(open(os.path.join(FS_DATASET_ROOT, "artifacts", \
            "yelp", "gen_summs", "{}.json".format(split))))
        for k1 in f:
            if k1[0] == '<':
                continue
            for k2 in f[k1]:
                refs = [" ".join(ref) for ref in f[k1][k2]['gold_summs'][0]]
                # summ = " ".join(f[k1][k2]['gen_summ'][0])
                summ = summaries[k2]
                with open(sum_file, 'w+') as fx:
                    fx.write("\n".join(sent_tokenize(summ)))
                    fx.flush()
                for i, ref in enumerate(refs):
                    with open(ref_files[i], 'w+') as fx:
                        fx.write("\n".join(sent_tokenize(ref)))
                        fx.flush()
                os.system(command)
                reader = csv.reader(open(csv_file))    
                rows = [row for row in reader]
                name = rows[1][0]
                rouge_for_name.append(float(rows[1][5]))
    
    print(name, ":", sum(rouge_for_name)/len(rouge_for_name))
    