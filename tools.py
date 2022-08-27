# -*- coding: utf-8 -*-
import random

#from sympy import true

def shuffle(lol,seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def minibatches(inputs=None,targets=None,batch_size=None):
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")

    for start_idx in range(0,len(inputs),batch_size):
        end_idx = start_idx + batch_size
        if end_idx>len(inputs):
            break

        excerpt = slice(start_idx,end_idx)

        yield inputs[excerpt],targets[excerpt]


def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win//2 * [0] + l + win//2 * [0]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def contextwin_2(ls,win):
    assert (win % 2) == 1
    assert win >=1
    outs=[]
    for l in ls:
        outs.append(contextwin(l,win))
    return outs

def getKeyphraseList(l):
    res, now= [], []
    for i in range(len(l)):
        if l[i] != 0:
            now.append(str(i))
        if l[i] == 0 or i == len(l) - 1:
            if len(now) != 0:
                res.append(' '.join(now))
            now = []
    return set(res)

def conlleval(predictions, groundtruth):
    assert len(predictions) == len(groundtruth)
    res = {}
    all_cnt, good_cnt = len(predictions), 0
    p_cnt, r_cnt, pr_cnt = 0, 0, 0
    for i in range(all_cnt):
        # print i
        if predictions[i][0:len(groundtruth[i])] == groundtruth[i]:
            good_cnt += 1
        pKeyphraseList = getKeyphraseList(predictions[i][0:len(groundtruth[i])])
        gKeyphraseList = getKeyphraseList(groundtruth[i])
        p_cnt += len(pKeyphraseList)
        r_cnt += len(gKeyphraseList)
        pr_cnt += len(pKeyphraseList & gKeyphraseList)
    res['a'] = 1.0*good_cnt/all_cnt
    res['p'] = 1.0*good_cnt/(p_cnt+0.1)
    res['r'] = 1.0*good_cnt/(r_cnt+0.1)
    res['f'] = 2.0*res['p']*res['r']/(res['p']+res['r']+0.1)
    return res


    