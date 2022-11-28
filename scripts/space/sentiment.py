from globals import *
from data import get_aspect, cosine_similarity, read_space_data

import os
import pickle
from nltk.tokenize import sent_tokenize

from pyabsa.functional import ATEPCCheckpointManager

aspect_extractor = None

def get_aspect_sentiment_pairs(sentences):
    global aspect_extractor
    if aspect_extractor is None:
        aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english')
    result = aspect_extractor.extract_aspect(inference_source=sentences,
            save_result=False,
            print_result=False,  # print the result
            pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
            )
    return result

def get_positive_negative_nums(sentences, aspect, combine=False):
    sentencewise_pairs = get_aspect_sentiment_pairs(sentences)
    n_pos, n_neg, n_tot = 0, 0, len(sentences)
    for line in sentencewise_pairs:
        p, n = 0, 0
        for (topic, sentiment) in zip(line['aspect'], line['sentiment']):
            if get_aspect(topic) == aspect:
                if combine:
                    n_pos += int(sentiment == "Positive")
                    n_neg += int(sentiment == "Negative") 
                else:
                    p += int(sentiment == "Positive")
                    n += int(sentiment == "Negative")
        if p > n:
            n_pos += 1
        elif n > p:
            n_neg += 1
    return (n_pos, n_neg, n_tot)

def get_all_sentiments_with_counts(sentences):
    result = {}
    sentencewise_pairs = get_aspect_sentiment_pairs(sentences)
    for line in sentencewise_pairs:
        for (topic, sentiment) in zip(line['aspect'], line['sentiment']):
            if sentiment == "Neutral":
                continue
            if topic not in result:
                result[topic] = [0,0]
            if sentiment == "Positive":
                result[topic][0] += 1
            elif sentiment == "Negative":
                result[topic][1] += 1
    return result

def sentiment_count_wrapper(sentences, aspect=None):
    sentiments = get_all_sentiments_with_counts(sentences)
    sentiments = [(word, sentiments[word][0], sentiments[word][1]) for \
        word in sentiments]
    if aspect is not None:
        # Re-order the sentiments according to decreasing cosine similarity
        neg_glove_similarities = [-cosine_similarity(s[0], aspect) for s in sentiments]
        sentiments = [s for _, s in sorted(zip(neg_glove_similarities, sentiments))]
    return sentiments        

def dump_sentiments_with_counts(summaries, dir="sentiments"):
    entities = read_space_data()[:10]
    topic_annotated = pickle.load(open(os.path.join(SPACE_SAVE_DATA_ROOT, \
        "topic-annotated.pkl"), 'rb'))
    root = os.path.join(SPACE_SAVE_DATA_ROOT, dir)
    os.makedirs(root, exist_ok=True)
    for entity in entities:
        eid = entity['entity_id']
        summary = summaries[eid]
        entity_dir = os.path.join(root, str(eid))
        if not os.path.exists(entity_dir):
            os.mkdir(entity_dir)
        sentences = {aspect:[] for aspect in aspects_[1:]}
        for (topics, aspects, sentence) in topic_annotated[eid]:
            for aspect in aspects:
                if aspect != "none":
                    sentences[aspect].append(sentence)
        for aspect in summary:
            if aspect == "none":
                continue
            output = []
            out_file = os.path.join(entity_dir, "{}.txt".format(aspect))
            output.append("***** {} : Sentiments for original sentences *****".format(aspect))
            sentiments = sentiment_count_wrapper(sentences[aspect])
            for sentiment in sentiments:
                output.append("{} : + {}     - {}".format(*sentiment))
            s = summary[aspect]
            if len(s) > 1 and type(s[0]) == list:
                output.append("***** {} : Sentiments for raw summaries *****".format(aspect))
                lines = sent_tokenize(" ".join(s[0]))
                sentiments = sentiment_count_wrapper(lines)
                for sentiment in sentiments:
                    output.append("{} : + {}     - {}".format(*sentiment))
                s = s[-1]
            elif len(s) == 1 and type(s) == list:
                s = s[0]
            output.append("***** {} : Sentiments for final summary *****".format(aspect))
            lines = sent_tokenize(" ".join(s))
            sentiments = sentiment_count_wrapper(lines)
            for sentiment in sentiments:
                output.append("{} : + {}     - {}".format(*sentiment))
            with open(out_file, "w+") as f:
                f.write("\n".join(output))

def get_aspectwise_pos_neg(tas):
    mapping = {}
    for eid in tas:
        sentences = {aspect:[] for aspect in aspects_[1:]}
        sentiment = {}
        for (topics, aspects, sentence) in tas[eid]:
            for aspect in aspects:
                if aspect != "none":
                    sentences[aspect].append(sentence)
        for aspect in sentences:
            sentiment[aspect] = get_positive_negative_nums(sentences[aspect], \
                aspect)
        mapping[eid] = sentiment 
    return mapping

def test():
    summaries = pickle.load(open(os.path.join(SPACE_SAVE_DATA_ROOT, \
        "review-stratified-aspectwise-summary.pkl"), 'rb'))
    dump_sentiments_with_counts(summaries, dir="sentiments/review-stratified")
    summaries = pickle.load(open(os.path.join(SPACE_SAVE_DATA_ROOT, \
        "long-noclusters-after-gpt3.pkl"), 'rb'))
    formatted = {eid  :{a[0] : a[1] for a in summaries[eid]} for eid in summaries}
    dump_sentiments_with_counts(formatted, dir="sentiments/qfsumm-gpt3-noclusters")

if __name__ == '__main__':
    test()