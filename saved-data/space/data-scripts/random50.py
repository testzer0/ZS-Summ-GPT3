import os
import pickle
import nltk
from nltk.tokenize import sent_tokenize
import random
import numpy as np
from functools import cmp_to_key

def compare(item1, item2):
    if item1[0] < item2[0]:
        return -1
    elif item1[0] > item2[0]:
        return 1
    elif item1[1] < item2[1]:
        return -1
    elif item1[1] > item2[1]:
        return 1
    elif item1[2] < item2[2]:
        return -1
    elif item1[2] > item2[2]:
        return 1
    else:
        return 0

done = []
random_draws = []

hs = pickle.load(open("summaries-pkl/gpt3-human-simplified.pkl", 'rb'))
eids = list(hs.keys())
aspects = list(hs['100597'].keys())

random.seed(2006)

while len(random_draws) < 50:
    eid = eids[random.randint(0, len(eids)-1)]
    aspect = aspects[random.randint(0, len(aspects)-1)]
    stmts = hs[eid][aspect]
    ind = random.randint(0, len(stmts)-1)
    stmt = stmts[ind]
    while stmt in done:
        ind = random.randint(0, len(stmts)-1)
        stmt = stmts[ind]
    random_draws.append((eid, aspect, ind+1, stmt))
    done.append(stmt)

random_draws = sorted(random_draws, key=cmp_to_key(compare))
with open("data-scripts/out.txt", "w+") as f:
    for eid, aspect, num, stmt in random_draws:
        f.write("{} {} {}    {}\n".format(eid, aspect, num, stmt))