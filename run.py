import datetime
import csv
import numpy as np
import os
import pandas as pd
import sys
import gc

import random
import os.path
from os import path

from datetime import datetime, timedelta
from collections import OrderedDict

DATA_DIR = '//'  # use your path
df = pd.read_csv(DATA_DIR + 'ENC1.csv', index_col=False, encoding="ISO-8859-1", low_memory=False)
print(df.columns)
df2 = pd.read_csv(DATA_DIR + 'ENC2.csv', index_col=False, encoding="ISO-8859-1", low_memory=False)
print(df2.columns)
df = df.append(df2)

STUDY_IDs = df.STUDY_ID.unique()
count = 0
import nltk
# nltk.download('punkt')

from nltk import tokenize

final_list = []
count = 0
print("enter")
count = 0
df1 = pd.read_csv('//RDoC_Dictionary_subset.csv')  # use sentence dictionary provided: RDoC_Dictionary_subset.csv
keyword_col = df1["Keyword"]
domain_col = df1["Domain"]
sentence_col = df1["Sentences"]
keywords = {}
count = 0
for keyword in keyword_col:
    if keyword not in keywords:
        keywords[keyword] = [count]
    else:
        mylist = keywords[keyword]
        mylist.append(count)
        keywords[keyword] = mylist
    count = count + 1
# print(keywords)
uniquekeys = list(keywords.keys())
print(uniquekeys)

import time

start = time.time()
count = 0
final_list = []
try:
    import cPickle as pickle
except:
    import pickle

print("enter")
ibegin = 0
iend = len(STUDY_IDs) / 4
for STUDY_ID in STUDY_IDs:
    count = count + 1
    if count > iend:
        continue
    if count < ibegin:
        continue
    if count % 2000 == 0:
        print('current patient count=' + str(count))

        end = time.time()
        print(end - start)

    df_sub = df.query('STUDY_ID==@STUDY_ID')
    df_sub.sort_values(['STUDY_ID', 'VISIT_ID', 'STUDY_NOTE_ID', 'LINE_NUM'], ascending=[True, True, True, True],
                       inplace=True)

    sentence_list_1 = []
    sentence_list_2 = []
    keyword_list = []
    domain_list = []
    STUDY_NOTE_ID_list = []
    CONTACT_DATE_list = []

    STUDY_NOTE_IDs = df_sub.STUDY_NOTE_ID.unique()
    lines_col = []
    STUDY_NOTE_ID_col = []
    CONTACT_DATE_col = []
    for STUDY_NOTE_ID in STUDY_NOTE_IDs:
        df_sub_sub = df_sub.query('STUDY_NOTE_ID==@STUDY_NOTE_ID')
        CONTACT_DATE = df_sub_sub.iloc[0]['CONTACT_DATE']
        lines = df_sub_sub['NOTE_TEXT'].to_string(index=False)
        if lines not in lines_col:
            lines_col.append(lines)
            STUDY_NOTE_ID_col.append(STUDY_NOTE_ID)
            CONTACT_DATE_col.append(CONTACT_DATE)
    linecount = 0
    for lines in lines_col:
        sentence_list = tokenize.sent_tokenize(lines)
        for i in range(len(uniquekeys)):
            keyword = str(uniquekeys[i])
            if lines.find(keyword) > 0:
                for sentence in sentence_list:
                    sentence = str(sentence)
                    if sentence.find(keyword) > 0:
                        for index in keywords[keyword]:
                            sentence_list_1.append(sentence)
                            sentence_list_2.append(sentence_col[index])
                            keyword_list.append(keyword)
                            domain_list.append(domain_col[index])
                            STUDY_NOTE_ID_list.append(STUDY_NOTE_ID_col[linecount])
                            CONTACT_DATE_list.append(CONTACT_DATE_col[linecount])
        linecount = linecount + 1
    if len(sentence_list_1) > 0:
        batch_size = 1000
        embedding_1 = model.encode(sentence_list_1, convert_to_tensor=True, batch_size=batch_size)
        embedding_2 = model.encode(sentence_list_2, convert_to_tensor=True, batch_size=batch_size)

        # embedding_1 = torch.tensor(model.encode_multi_process(sentence_list_1, batch_size=batch_size, pool=model.start_multi_process_pool(target_devices=["cuda:0","cuda:1"]))).to(device=device)
        # embedding_2 = torch.tensor(model.encode_multi_process(sentence_list_2, batch_size=batch_size, pool=model.start_multi_process_pool(target_devices=["cuda:0","cuda:1"]))).to(device=device)
        a_norm = torch.nn.functional.normalize(embedding_1, p=2, dim=-1)
        b_norm = torch.nn.functional.normalize(embedding_2, p=2, dim=-1)
        val = torch.sum(a_norm * b_norm, dim=-1)
        threshold = 0.3
        ids = (((val >= threshold).to(torch.int16)).nonzero(as_tuple=True))[0]
        if len(ids) > 0:
            for id in ids:
                final_list.append(
                    [STUDY_ID, STUDY_NOTE_ID_list[id], CONTACT_DATE_list[id], keyword_list[id], domain_list[id],
                     float(val[id])])

with open("//result.pkl", 'wb') as f:
    pickle.dump(final_list, f)


