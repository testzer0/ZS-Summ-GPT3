from itertools import count
import os
import pickle
import json
import krippendorff

import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

# names = ['TCG', 'TQG', 'AceSum', 'QFSumm']
axes = ['Intrinsic Factuality', 'Extrinsic Factuality', 'Faithfulness', \
    'Relevance']
names = ['TCG', 'TQG', 'QG', 'AceSum', 'QFSumm']
maxscores = [3, 3, 5, 5]
persons = ['adithya', 'greg', 'alex']


scores = json.load(open('scores-space.json'))


def save_scores():
    for name in scores:
        for row in range(len(scores[name][0][0])):
            p = perm[row]
            psort = sorted(p)
            m = [p.index(v) for v in psort]
            for met in range(len(scores[name])):
                temp = [scores[name][met][m[i]][row] for i in range(len(scores[name][met]))]
                for i in range(len(scores[name][met])):
                    scores[name][met][i][row] = temp[i]
    scores2 = {}
    for name in scores:
        scores2n = {axes[i] : scores[name][i] for i in range(len(axes))}
        scores2[name] = scores2n
        
    json.dump(scores2, open('scores-space.json', 'w+'))

def get_averages(axis):
    avgs = 0
    for person in persons:
        ss = np.array(scores[person][axis])
        ss = np.mean(ss, axis=1)
        avgs = avgs + ss
    avgs /= len(persons)
    return avgs.tolist()

def get_krippendorfs_alpha(axis):
    ss = []
    for person in persons:
        ss.append(np.ravel(np.array(scores[person][axis])))
    ss = np.array(ss)
    ka = krippendorff.alpha(reliability_data=ss)
    return ka

def get_fleiss_kappa(axis):
    ss = []
    maxscore = maxscores[axes.index(axis)]
    for person in persons:
        ss.append(np.ravel(np.array(scores[person][axis])))
    ss = np.array(ss)
    count_array = np.zeros((ss.shape[1], maxscore))
    for j in range(maxscore):
        count_array[:,j] = np.sum((ss == j+1).astype(int), axis=0)
    return fleiss_kappa(count_array, method='fleiss')

if __name__ == '__main__':
    for axis in axes:
        avgs = [str(int(s*100)/100) for s in get_averages(axis)]
        print(axis + " - " + " ".join(avgs))
        # ka = get_krippendorfs_alpha(axis)
        # ka = int(10000*ka)/10000
        # print(axis + " - ", ka)
        # fk = get_fleiss_kappa(axis)
        # fk = int(10000*fk)/10000
        # print(axis + " - ", fk)
