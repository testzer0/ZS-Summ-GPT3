import os
import pickle

def collect_summaries(root_dir):
    summarized = {}
    eids = os.listdir(root_dir)
    for eid in eids:
        entity_dir = os.path.join(root_dir, eid)
        summaries = {}
        for fname in os.listdir(entity_dir):
            aspect = fname.split('.')[0]
            with open(os.path.join(entity_dir, fname)) as f:
                summaries[aspect] = [line.strip() for line in f]
        summarized[eid] = summaries
    return summarized

if __name__ == '__main__':
    simplified = collect_summaries("gpt3-summarized-new")
    pickle.dump(simplified, open("gpt3-human-simplified.pkl", 'wb+'))            