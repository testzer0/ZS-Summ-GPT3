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
from kneed import KneeLocator
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import matplotlib.pyplot as plt

# from model_summac import SummaCConv, SummaCZS
# from simcse import SimCSE

model = None
lemmatizer = None 
nltk_stopwords = None

DS_DIR = os.path.join(PROMPT_DIR[:PROMPT_DIR.rfind("/")], "baselines/DiscourseSimplification")
BISECT_DIR = os.path.join(PROMPT_DIR[:PROMPT_DIR.rfind("/")], "baselines/BiSECT/")

def simplify_sentence(sentence):
    prompt = \
    """
    Simplify the given sentences into simple value judgements.
    Sentence: Some reviewers found the rooms to be dusty.
    Output: The rooms were dusty.
    Sentence: The heater was found to not function properly.
    Output: The heater did not function properly.
    Sentence: The hotel has a stain on the carpet.
    Output: There is a stain on the carpet.
    Sentence: Some guests were unhappy with the cost of the parking.
    Output: The cost of parking was too high.
    Sentence: {}
    Output:
    """.format(sentence)
    return get_gpt3_response(prompt).strip()

def simplify_split_summaries(summarized):
    simplified = {}
    for eid in summarized:
        simplified_for_eid = {}
        for aspect in summarized[eid]:
            simplified_for_aspect = []
            for (sentence, parts) in summarized[eid][aspect]:
                simplified_parts = [simplify_sentence(part) for part in parts]
                simplified_for_aspect.append((sentence, simplified_parts))
            simplified_for_eid[aspect] = simplified_for_aspect
            print("Aspect {} for eid {} done.".format(aspect, eid))
        simplified[eid] = simplified_for_eid
    return simplified

def split_and_rephrase_gpt3(sentence):
    prompt = \
    """
    Split the given statement into simple propositions:
    Sentence: 
    The hotel is generally clean, with the exception of a stain on the carpet.
    Propositions:
    The hotel is clean.
    The hotel has a stain on the carpet.
    Sentence:
    {}
    Propositions:
    """.format(sentence)
    response = get_gpt3_response(prompt)
    propositions = [proposition.strip() for proposition in response.split('\n') if \
        proposition.strip() != '']
    return propositions

def split_and_rephrase_gpt3_for_summarized(summarized, summaries_are_lists=False, \
    summary_type='initial', join=True):
    if type(summarized) == str:
        summarized = pickle.load(open(\
            os.path.join(SPACE_SAVE_DATA_ROOT, summarized), 'rb'))
    result = {}
    for eid in summarized:
        proposition_map = {}
        summaries = summarized[eid]
        if type(summaries) == list:
            summaries = {a[0]:a[1] for a in summaries}
        for aspect in summaries:
            summary = summaries[aspect]
            if type(summary) == list:
                if summaries_are_lists:
                    summary = " ".join(summary)
                elif summary_type == 'initial':
                    if type(summary[0]) == list:
                        summary = " ".join(summary[0])
                    else:
                        summary = summary[0]
                else:
                    summary = summary[-1]
                    if type(summary) == list:
                        summary = " ".join(summary)
            propositions = []
            for sent in sent_tokenize(summary):
                if join:
                    propositions += split_and_rephrase_gpt3(sent)
                else:
                    sr = split_and_rephrase_gpt3(sent)
                    propositions.append((sent, sr))
            proposition_map[aspect] = propositions
            print("Aspect {} for eid {} done.".format(aspect, eid))
        result[eid] = proposition_map
    return result

def is_simple_sentence(sentence):
    words = word_tokenize(sentence)
    for word in ['while', 'but', 'though', 'although', 'other', 'others', \
        'however']:
        if word in words:
            return False
    return True

def get_num_simple_and_complex_sents(summarized, normalized=False, is_in_list=False, \
    split_or_simplified=False, to_summarize='initial'):
    n_simple, n_complex = 0, 0
    for eid in summarized:
        summaries = summarized[eid]
        if type(summaries) == list:
            summaries = {s[0] : s[1] for s in summaries}
        for aspect in aspects_[1:]:
            summary = summaries[aspect]
            if is_in_list:
                summary = " ".join(summary)
            elif split_or_simplified:
                joined = []
                for (parent, parts) in summary:
                    joined += parts
                summary = joined
            elif type(summary) == list:
                if to_summarize == 'initial':
                    if type(summary[0]) == list:
                        summary = " ".join(summary[0])
                    else:
                        summary = summary[0]
                else:
                    summary = summary[-1]
            for sent in sent_tokenize(summary):
                if is_simple_sentence(sent):
                    n_simple += 1
                else:
                    n_complex += 1
    if normalized and n_simple+n_complex > 0:
        n_simple, n_complex = 1.0*n_simple/(n_simple+n_complex), \
            1.0*n_complex/(n_simple+n_complex)
    return n_simple, n_complex

def simple_keyword_filters(keywords):
    return [keyword for keyword in keywords if keyword.isalpha()]

def string_found(string1, string2):
    # https://stackoverflow.com/questions/4154961/find-substring-in-string-but-only-if-whole-words
    if re.search(r"\b" + re.escape(string1) + r"\b", string2):
        return True
    return False

def get_lowercase_lemmatized_words(sentence):
    global lemmatizer
    if lemmatizer is None:
        lemmatizer = nltk.wordnet.WordNetLemmatizer()
    sentence = word_tokenize(sentence.lower())
    return [lemmatizer.lemmatize(word) for word in sentence]

def stem_lowercase_and_remove_stopwords(sentence):
    global lemmatizer, nltk_stopwords
    if lemmatizer is None:
        lemmatizer = nltk.wordnet.WordNetLemmatizer()
    if nltk_stopwords is None:
        nltk_stopwords = stopwords.words('english')
    sentence = word_tokenize(sentence.lower())
    words = [lemmatizer.lemmatize(word) for word in sentence if word not in nltk_stopwords]
    return " ".join(words)

def tf_idf(src_stmts, tf_stmts=None, max_keywords=-1, preprocess=True):
    global nltk_stopwords
    if type(src_stmts) == str:
        src_stmts = sent_tokenize(src_stmts)
    if tf_stmts is None:
        tf_stmts = src_stmts
    elif type(tf_stmts) == str:
        tf_stmts = sent_tokenize(tf_stmts)
    if preprocess:
        src_stmts = [stem_lowercase_and_remove_stopwords(src_stmt) for src_stmt in src_stmts]
        tf_stmts = [stem_lowercase_and_remove_stopwords(tf_stmt) for tf_stmt in tf_stmts]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(src_stmts)
    names = vectorizer.get_feature_names()
    keywords = []
    scores = []
    for tf_stmt in tf_stmts:
        response = vectorizer.transform([tf_stmt])
        sorted_nzs = np.argsort(response.data)[:-(max_keywords+1):-1].tolist()
        indices = response.indices.tolist()
        k = [names[indices[i]] for i in sorted_nzs]
        s = response.data.tolist()
        s = [s[i] for i in sorted_nzs]
        keywords.append(k)
        scores.append(s)
    return keywords, scores

def get_top_k_statements_based_on_keywords(sentence, keywords, references, max_keywords=5, \
    k=5, weights=None, return_scores=False):
    if type(references) == str:
        references = sent_tokenize(references)
    if weights is None:
        weights = [1]*len(keywords)
    wordized_refs = [get_lowercase_lemmatized_words(s) for s in references]
    wordized_sent = get_lowercase_lemmatized_words(sentence)
    keywords_with_weights = [(keyword.lower(), weight) for keyword, weight in \
        zip(keywords, weights) if keyword.lower() in wordized_sent][:max_keywords]
    sentences_with_scores = [(s, sum([ref.count(kw[0])*kw[1] for \
        kw in keywords_with_weights])) for s,ref in zip(references, wordized_refs)]
    sentences_with_scores = sorted(sentences_with_scores, key=lambda x : -x[1])[:k]
    if not return_scores:
        return [sws[0] for sws in sentences_with_scores]
    return sentences_with_scores

def get_model(type="conv"):
    if type.lower() == "conv":
        return SummaCConv(granularity="sentence")
    elif type.lower() == "simcse":
        return SimCSE("princeton-nlp/sup-simcse-roberta-large")
    else:
        return SummaCZS(granularity="sentence", model_name="vitc")    

def get_simplified_statements(sentence):
    with open(os.path.join(DS_DIR, "input.txt"), 'w+') as f:
        f.write(sentence)
    os.system("cd {} && mvn clean compile exec:java".format(DS_DIR))
    sentences = []
    for line in open(os.path.join(DS_DIR, "output_default.txt"), 'r'):
        if line[0] == '\t' or line.strip() == '' or line.strip()[0] == '#':
            continue
        line = line.strip().split('\t')
        sentences.append(" ".join(line[2:]))
    return sentences
    
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

def get_top_supporting_and_weakening_lines(sentences, summary, k=5, model_type="zs", \
    simplify=False, filter_len=True):
    global model
    if model is None:
        model = get_model(model_type)
    if type(summary) == str:
        summary = sent_tokenize(summary)
        if filter_len:
            summary = [line.strip() for line in summary if len(line.strip().split(" ")) >= 4]
        else:
            summary = [line.strip() for line in summary]
    if simplify:
        s = []
        for sentence in summary:
            s += get_simplified_statements(sentence)
        summary = s
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
    overall_score = sum(linewise_scores)*1.0/len(linewise_scores)
    return result, linewise_scores, overall_score

def save_supporting_and_weakening_statements(topicwise_annotated, summarized, \
    root=SPACE_SAVE_DATA_ROOT, dir_name="entailment", model_type="zs", \
    to_summarize="initial", verbose=False, is_in_list=False, filter_len=True, \
    split_or_simplified=False, have_all_statements=False):
    aspectwise_statements = get_aspectwise_statements(topicwise_annotated, \
        have_all_statements)
    root_dir = os.path.join(root, dir_name)
    max_pos = []
    max_neg = []
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    for eid in aspectwise_statements:
        entity_dir = os.path.join(root_dir, eid)
        if not os.path.exists(entity_dir):
            os.mkdir(entity_dir)
        summaries = summarized[eid]
        if type(summaries) == list:
            summaries = {s[0] : s[1] for s in summaries}
        for aspect in aspects_[1:]:
            out_file = os.path.join(entity_dir, "{}.txt".format(aspect))
            sentences = aspectwise_statements[eid][aspect]
            summary = summaries[aspect]
            parents = None
            if is_in_list:
                summary = " ".join(summary)
            elif split_or_simplified:
                parents = []
                joined = []
                for (parent, parts) in summary:
                    for part in parts:
                        parents.append(parent)
                        joined.append(part)
                summary = joined
            elif type(summary) == list:
                if to_summarize == 'initial':
                    if type(summary[0]) == list:
                        summary = " ".join(summary[0])
                    else:
                        summary = summary[0]
                else:
                    summary = summary[-1]
            top_supporting_and_weakening_lines, linewise_scores, overall_score = \
                get_top_supporting_and_weakening_lines(sentences, summary, \
                model_type=model_type, simplify=False, filter_len=filter_len)
            output = ["***** Aspect: {} Model Type : {} *****".format(aspect, model_type)]
            output.append("Overall score: {}".format(overall_score))
            if not parents:
                parents = [None]*len(linewise_scores)
            for parent, line_score, (line, supporting, weakening) in zip( \
                parents, linewise_scores, top_supporting_and_weakening_lines):
                output.append("")
                output.append("**********")
                output.append("")
                output.append("[{}] {}".format(line_score, line))
                if parent is not None:
                    output.append("Parent: {}".format(parent))
                output.append("")
                output.append("----- Supporting -----")
                output.append("")
                for line, score in supporting:
                    output.append("[{}] {}".format(score, line))
                output.append("")
                output.append("----- Weakening -----")
                output.append("")
                for line, score in weakening:
                    output.append("[{}] {}".format(score, line))
                max_pos.append(supporting[0][1])
                max_neg.append(weakening[0][1])
            with open(out_file, 'w+') as f:
                f.write("\n".join(output))
            if verbose:
                print("Aspect {} for eid {} done.".format(aspect, eid))
    plt.scatter(max_pos, max_neg, marker='x', c='red')
    plt.xlabel("Maximum positive score")
    plt.ylabel("Maximum negative score")
    plt.title("Max positive v/s Max negative scores")
    plt.savefig(os.path.join(root_dir, "scatter.png"), bbox_inches='tight')
    
def return_supporting_and_weakening_statements(space, summarized, model_type="zs", \
    filter_len=True):
    ret = {}
    for eid in summarized:
        eidwise = {}
        sentences = []
        for review in space[eid]['reviews']:
            sentences += review['sentences']
        for aspect in summarized[eid]:
            name = "temp/old9/{}-{}.pkl".format(eid, aspect)
            if os.path.exists(name):
                eidwise[aspect] = pickle.load(open(name, 'rb'))
                print("Found {}.".format(name[:-4]))
                continue
            summary = summarized[eid][aspect]
            parents = None
            parents = []
            joined = []
            for (parent, parts) in summary:
                for part in parts:
                    parents.append(parent)
                    joined.append(part)
            summary = joined
            top_supporting_and_weakening_lines, linewise_scores, _ = \
                get_top_supporting_and_weakening_lines(sentences, summary, \
                model_type=model_type, simplify=False, filter_len=filter_len)
            tosave = []
            for parent, line_score, (line, supporting, weakening) in zip( \
                parents, linewise_scores, top_supporting_and_weakening_lines):
                tosave.append((line_score, line, parent, supporting, weakening))
            eidwise[aspect] = tosave
            pickle.dump(tosave, open(name, 'wb+'))
            print("{} done.".format(name[:-4]))
        ret[eid] = eidwise
        print("EID {} done.".format(eid))
    os.system("rm temp/old9/*")
    return ret

def save_entailment_scores(summary_file_name, dir_name, model_type="zs", is_in_list=False, \
    filter_len=False, return_instead_of_save=False, split_or_simplified=False, \
    have_all_statements=False):
    topicwise_annotated = pickle.load(open(os.path.join(\
        SPACE_SAVE_DATA_ROOT, "topic-annotated.pkl"), 'rb'))
    summarized = pickle.load(open(os.path.join(SPACE_SAVE_DATA_ROOT, summary_file_name), 'rb'))
    if return_instead_of_save:
        return return_supporting_and_weakening_statements(topicwise_annotated, summarized, \
            verbose=True, is_in_list=is_in_list, filter_len=filter_len, \
            split_or_simplified=split_or_simplified, have_all_statements=have_all_statements)
    else:
        save_supporting_and_weakening_statements(topicwise_annotated, summarized, \
            dir_name=dir_name, model_type=model_type, verbose=True, is_in_list=is_in_list, \
            filter_len=filter_len, split_or_simplified=split_or_simplified, \
            have_all_statements=have_all_statements)

def save_top_k_tfidf_statements(summary_file_name, out_dir, reference="reviews", k=5, \
    summary_type="initial", equal_weight_to_keywords=False, summaries_are_lists=False):
    topicwise_annotated = pickle.load(open(os.path.join(\
        SPACE_SAVE_DATA_ROOT, "topic-annotated.pkl"), 'rb')) if \
        reference == "reviews" else None
    summarized = pickle.load(open(os.path.join(SPACE_SAVE_DATA_ROOT, \
        summary_file_name), 'rb'))
    root_dir = os.path.join(SPACE_SAVE_DATA_ROOT, out_dir)
    os.makedirs(root_dir, exist_ok=True)
    for eid in summarized:
        eid_dir = os.path.join(root_dir, eid)
        if not os.path.exists(eid_dir):
            os.mkdir(eid_dir)
        summaries = summarized[eid]
        if type(summaries) == list:
            summaries = {s[0] : s[1] for s in summaries}
        for aspect in summaries:
            summary = summaries[aspect]
            if type(summary) == list:
                if summaries_are_lists:
                    summary = " ".join(summary)
                elif summary_type == 'initial':
                    if type(summary[0]) == list:
                        summary = " ".join(summary[0])
                    else:
                        summary = summary[0]
                else:
                    summary = summary[-1]
            summaries[aspect] = summary
        refs = {aspect: [] for aspect in aspects_[1:]}
        for (topics, aspects, ref) in topicwise_annotated[eid]:
            for aspect in aspects:
                if aspect != 'none':
                    refs[aspect].append(ref)
        if reference == "reviews":
            stmts = refs
        else:
            stmts = summaries
        for aspect in summaries:
            output = []
            output.append("***** {} - {} *****".format(eid, aspect))
            output.append("")
            sents = sent_tokenize(summaries[aspect])
            keywords, scores = tf_idf(stmts[aspect], sents)
            top_k_sents = [(sent, get_top_k_statements_based_on_keywords(sent, kws, refs[aspect], \
                max_keywords=1000, k=k, weights=(None if equal_weight_to_keywords else ss))) for \
                sent, kws, ss in zip(sents, keywords, scores)]
            first = True
            for (sent, top_k), kws, ss in zip(top_k_sents, keywords, scores):
                if first:
                    first = False
                else:
                    output.append("")
                    output.append("----------")
                    output.append("")
                output.append(sent)
                output.append("Keywords: ")
                for kw, s in zip(kws, ss):
                    output.append("     [{}]  {}".format(s, kw))
                output.append("")
                output.append("Top {} statements based on keywords:".format(k))
                for i, top in enumerate(top_k):
                    output.append("{}:  ".format(i+1)+top)
            with open(os.path.join(eid_dir, "{}.txt".format(aspect)), "w+") as f:
                f.write("\n".join(output))
            print("Aspect {} for eid {} done.".format(aspect, eid))

def similarity_score(sentences1, sentences2, model_type="zs", thresh=0.5):
    global model
    if model is None:
        model = get_model(model_type)
    if type(sentences1) == str:
        sentences1 = sent_tokenize(sentences1)
    if type(sentences2) == str:
        sentences2 = sent_tokenize(sentences2)
    avg_top = 0.0
    n_top = len(sentences2)
    count = 0
    for s2 in sentences2:
        if model_type.lower() == "simcse":
            scores = [model.similarity([s1], [s2])[0][0] \
                for s1 in sentences1]
        else:
            scores = [model.score([s1], [s2])["scores"][0] \
                for s1 in sentences1]
        max_score = max(scores)
        avg_top += max_score
        if max_score > thresh:
            count += 1
    return avg_top / n_top, count, n_top

def self_similarity(sentences, model_type="zs", thresh=0.5):
    global model
    if model is None:
        model = get_model(model_type)
    if type(sentences) == str:
        sentences = sent_tokenize(sentences)
    avg_top = 0.0
    n_top = len(sentences)
    count = 0
    print(n_top)
    for s2 in sentences:
        if model_type.lower() == "simcse":
            scores = [model.similarity([s1], [s2])[0][0] \
                for s1 in sentences if s1 != s2]
        else:
            scores = [model.score([s1], [s2])["scores"][0] \
                for s1 in sentences if s1 != s2]
        max_score = max(scores)
        avg_top += max_score
        if max_score > thresh:
            count += 1
    return avg_top / n_top, count, n_top

def format_summaries(summarized, summaries_are_lists=False, \
    summary_type='initial', split_or_rephrased=False):
    for eid in summarized:
        summaries = summarized[eid]
        if type(summaries) == list:
            summaries = {a[0]:a[1] for a in summaries}
        for aspect in summaries:
            summary = summaries[aspect]
            if type(summary) == list:
                if summaries_are_lists:
                    summary = " ".join(summary)
                elif split_or_rephrased:
                    s = []
                    for (_, parts) in summary:
                        s += parts
                    summary = " ".join(parts)
                elif summary_type == 'initial':
                    if type(summary[0]) == list:
                        summary = " ".join(summary[0])
                    else:
                        summary = summary[0]
                else:
                    summary = summary[-1]
            summaries[aspect] = summary
        summarized[eid] = summaries
    return summarized

def similarity_score_across_hotels(summarized, thresh=0.5):
    total_score = 0.0
    fraction = 0.0
    n_score = 0
    n_fraction = 0
    for eid1 in summarized:
        for eid2 in summarized:
            if eid1 == eid2:
                continue
            summaries1 = summarized[eid1]
            summaries2 = summarized[eid2]
            for aspect in summaries1:
                summary1 = sum([s[1] for s in summaries1[aspect]], [])
                summary2 = sum([s[1] for s in summaries2[aspect]], [])
                score, f, nf = similarity_score(summary1, summary2, thresh=thresh)
                total_score += score
                fraction += f
                n_score += 1
                n_fraction += nf
    return total_score / n_score, fraction / n_fraction

def similarity_score_single_hotel(summarized, summaries_are_lists=False, \
    summary_type='initial', split_or_rephrased=False, thresh=0.5):
    summarized = format_summaries(summarized, summaries_are_lists, summary_type, \
        split_or_rephrased)
    total_score = 0.0
    fraction = 0.0
    n_score = 0
    n_fraction = 0
    for eid in summarized:
        summaries = summarized[eid]
        for aspect in summaries:
            summary = summaries[aspect]
            if len(summary) <= 1:
                continue
            score, f, nf = self_similarity(summary, thresh=thresh)
            total_score += score
            fraction += f
            n_score += 1
            n_fraction += nf
    return total_score / n_score, fraction / n_fraction

def read_entailment_file(fpath):
    ret = []
    with open(fpath) as f:
        lines = f.readlines()[5:]
        lines = [line.strip() for line in lines if line.strip() != '']
        i = 0
        while i < len(lines):
            statement = lines[i]
            statement = statement[statement.find(']')+2:].strip()
            supporting = []
            weakening = []
            while lines[i] != '----- Supporting -----':
                i += 1
            i += 1
            while lines[i] != '----- Weakening -----':
                ss = lines[i]
                pos = ss.find(']')
                score = float(ss[1:pos])
                ss = ss[pos+2:].strip()
                supporting.append((ss, score))
                i += 1
            i += 1
            while i < len(lines) and lines[i] != '**********':
                ws = lines[i]
                pos = ws.find(']')
                score = float(ws[1:pos])
                ws = ws[pos+2:].strip()
                weakening.append((ws, score))
                i += 1
            ret.append((statement, supporting, weakening))
            i += 1
    return ret

def average_top_score(ent_data):
    avg = 0
    n = 0
    for eid in ent_data:
        for aspect in ent_data[eid]:
            for (score,_,_,_,_) in ent_data[eid][aspect]:
                avg += score
                n += 1
    return avg/n

def support_percents(ent_data, thresh=0.75):
    counts = [0]*6
    for eid in ent_data:
        for aspect in ent_data[eid]:
            for (_, _, _, supp, _) in ent_data[eid][aspect]:
                s = [se[1] for se in supp if se[1] >= thresh]
                counts[len(s)] += 1
    counts = [100.0*c/sum(counts) for c in counts]
    return counts

def avg_length_in_words(summarized):
    total_length = 0
    n = 0
    for eid in summarized:
        for aspect in summarized[eid]:
            n += 1
            for sent in sent_tokenize(summarized[eid][aspect]):
                total_length += len(word_tokenize(sent))
    return total_length/n

if __name__ == '__main__':
    # save_top_k_tfidf_statements("summaries-pkl/gpt3-human-simplified-extracted.pkl", "tfidf/gpt3-hse", \
    #      summaries_are_lists=True)
    
    space = read_space_data()
    
    space = {entity['entity_id'] : entity for entity in space}
    
    # pkl_names = ['rg-first.pkl', 'tcg-first.pkl', 'rg.pkl']
    pkl_names = ['tcg.pkl', 'tqg.pkl', 'qg.pkl', 'acesum.pkl', 'qfsumm.pkl']
    sum_dir = os.path.join(SPACE_SAVE_DATA_ROOT, "all-new-pkls", "summaries-pkl")
    sr_dir = os.path.join(SPACE_SAVE_DATA_ROOT, "all-new-pkls", "sr-pkl")
    ent_dir = os.path.join(SPACE_SAVE_DATA_ROOT, "all-new-pkls", "entailment")

    high_scoring1 = json.load(open('temp/hs.json'))
    high_scoring2 = json.load(open('../fewsum/temp/hs.json'))
    # low_scoring = json.load(open('temp/hs.json'))
    
    perm1 = np.random.permutation(len(high_scoring1))[:50]
    perm2 = np.random.permutation(len(high_scoring2))[:50]
    high_scoring = [high_scoring1[i] for i in perm1] + \
        [high_scoring2[i] for i in perm2]
    # low_scoring = [low_scoring[i] for i in perm1]
    
    all_scoring = high_scoring # + low_scoring
    perm = np.random.permutation(len(all_scoring))
    all_scoring = [all_scoring[i] for i in perm]
    
    rownames = ['Score', 'Statement', 'Reference', 'Original Statement Before SR']
    rows = [rownames]
    for score, statement, parent, ref in all_scoring:
        rows.append([score, statement, ref, parent])

    with open('temp/hes.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)

    exit(0)
    for name in pkl_names:
        in_path = os.path.join(ent_dir, name)
        ent_data = pickle.load(open(in_path, 'rb'))
        high_scoring = []
        low_scoring = []
        
        for eid in ent_data:
            for aspect in aspects_[1:]:
                for _, statement, parent, supporting, weakening in \
                    ent_data[eid][aspect]:
                    for s, score in supporting:
                        if score > 0.5:
                            print(s)
                            high_scoring.append((score, statement, parent, s))
                    for s, score in weakening:
                        if score < -0.5:
                            print(s)
                            low_scoring.append((score, statement, parent, s))
        
    json.dump(high_scoring, open('temp/hs.json', 'w+'))
    json.dump(low_scoring, open('temp/ls.json', 'w+'))
    print(len(high_scoring))
    print(len(low_scoring))
            