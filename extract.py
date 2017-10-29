from pprint import pprint
import ast
import sys
import numpy as np
import pdb


def Read_Store(name,docs):

    # Read all lines into a list
    with open(name) as f:
        #content = f.readlines()
        content = f.readlines()
        #content = [next(f) for x in xrange(10*read)]

    # Convert each list item to a dict
    content = [ ast.literal_eval( line ) for line in content ]
    count=0
    for temp in content:
        t1=temp['title']
        if (t1[0]=='\'' and t1[-1]=='\'') or (t1[0]=='\"' and t1[-1]=='\"'):
            t1=t1[1:-1]
        #t2=(t1,)

        label = 0 if name == "normalnews.json" else 1
        docs.append((t1, label))
        count+=1

    return docs
