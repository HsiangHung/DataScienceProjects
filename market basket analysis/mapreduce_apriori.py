
from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext

import numpy as np
import copy
from time import time


threshold = 10


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: wordcount <file>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="Apriori_algorithm")
    lines = sc.textFile(sys.argv[1], 1)

    header = lines.take(1)[0]

    lines = lines.filter(lambda x: x!= header).map(lambda x: x.split(','))  

    records = lines.map(lambda x: (int(x[0]),[x[3]])).reduceByKey(lambda x,y: x+y)  
    ## (order1,[item1, item2..]), (order2, [item3, item4...]),...

    records = records.map(lambda x: (x[0],set(x[1])))
    ## (order1,set(item1, item2..)), (order2, set(item3, item4...)),....

    ## -------------------------------

#    filtered_items = lines.map(lambda x: (x[3], 1)).reduceByKey(lambda x,y: x+y).filter(lambda x: x[1] >= 10)

    pruned_itemsets = lines.map(lambda x: (x[3], [int(x[0])])).reduceByKey(lambda x,y: x+y).\
                     map(lambda x: (x[0], len(set(x[1])))).filter(lambda x: x[1] >= threshold)
    ## pruned_itemsets = (item1,[order1]),(item2,[order3]),(item1,[order1]),(item1,[order2]),...
    ##                 -> (item1, [order1, order1, order2]), (item2, [order3, order4, order5, order3]), ... 
    ##                    note: make set([]) is to avoid repeated counts
    ##                 -> (item1, 2), (item2, 3),....
    ##          filter -> (item2, 3) ..

    #print (purchase_times.map(lambda x: (x[1], x[0])).max())
    print (pruned_itemsets.take(4))
    print (pruned_itemsets.count())

    #print (records.take(5))    
    
    print (records.count())

    print (records.take(5))

    pruned_itemsets = set(pruned_itemsets.map(lambda x:x[0]).collect()) 
    ## now 'pruned_itemsets' is the 2-item pruned itemsets, i.e. absolute support > threshold
 

    print (type(pruned_itemsets))
    print ('new', len(pruned_itemsets))







    sc.stop()
