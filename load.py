# -*- coding: utf-8 -*-
import numpy as np
import pickle as cPickle
import random

def getrowdata(file_path):
    data_list,tag_list = [],[]
    with open(file_path,'r') as f:
        for line in f:
            line = line.strip()
            data_list.append(line.split('\t')[0])
            tag_list.append(line.split('\t')[1])
    return data_list,tag_list        


def atisfold():
    f = open('data/data_set.pkl','rb')
    train_set, test_set, dicts = cPickle.load(f)
    embedding = cPickle.load(open('data/embedding.pkl','rb'))
    return train_set, test_set,dicts,embedding

def pad_sentences(sentences, padding_word=0, forced_sequence_length=None):
    if forced_sequence_length is None:
        sequence_length=max(len(x) for x in sentences)
    else:
        sequence_length=forced_sequence_length
    padded_sentences=[]
    for i in range(len(sentences)):
        sentence=sentences[i]
        num_padding=sequence_length-len(sentence)
        if num_padding<0:
            padded_sentence=sentence[0:sequence_length]
        else:
            padded_sentence=sentence+[int(padding_word)]*num_padding

        padded_sentences.append(padded_sentence)

    return padded_sentences








