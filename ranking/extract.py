from pprint import pprint
import ast
import sys
import numpy as np
import pdb
import json


def Read_Store_Ranking_TRAIN():

    normal = '../normalnews.json' 
    weird = '../weirdnews.json'

    docs = []
    with open(normal) as f:
        content1 = f.readlines()
    with open(weird) as f:
        content2 = f.readlines()[500:]

    # Convert each list item to a dict
    content1 = [ ast.literal_eval( line ) for line in content1 ]
    for temp in content1:
        t1=temp['title']
        if (t1[0]=='\'' and t1[-1]=='\'') or (t1[0]=='\"' and t1[-1]=='\"'):
            t1=t1[1:-1]
        #t2=(t1,)

        label = 0
        docs.append((t1, label))
    
    content2 = [ ast.literal_eval( line ) for line in content2 ]
    for temp in content2:
        t1=temp['title']
        if (t1[0]=='\'' and t1[-1]=='\'') or (t1[0]=='\"' and t1[-1]=='\"'):
            t1=t1[1:-1]
        #t2=(t1,)

        label = 1
        docs.append((t1, label))
    return docs



def Read_Store_Ranking_TEST():
    annotate = ["annotations/weirdnews_annotated1.json", "annotations/weirdnews_annotated2.json", "annotations/weirdnews_annotated3.json"]
    files = [open(name) for name in annotate]
    data1, data2, data3 = files[0].readlines(), files[1].readlines(), files[2].readlines()
    for fil in files:
        fil.close()

    docs = []
    for i, line in enumerate(data1):
        docs.append([json.loads(line)['text'], 0])

    for data in [data1, data2, data3]:
        for i, line in enumerate(data):
            docs[i][1] += int(json.loads(line)['annotation'])
    for i in range(len(docs)):
        docs[i][1] = (docs[i][1] * 1.0)/9

    return docs
