import numpy as np
import operator
import copy

import pymysql as mdb

local_db = mdb.connect(user="root", host="localhost", db="basket_analysis", charset='utf8')


class Apriori_search(object):
    '''Search for itemsets whose support and confidence are beyond threshold'''

    def __init__(self, min_support, max_k_items):
        '''loading the data'''
        self.max_k = max_k_items
        self.min_support = min_support

        #hd  = open('/Users/hsianghung/Desktop/bi-dw/query_8_17.csv')
        hd  = open('query_results.csv')
        #hd  = open('interdrinks.csv')
        #hd  = open('beerhawk.csv')
        self.order_records = {}
        self.supports = {}

        self.sku_names = {}

        self.c_itemsets = set({})
        irow = -1
        for line in hd:
            irow += 1
            #if 5000 > irow > 0:
            if irow > 0:
                line = line.replace('\n','').split(',')
                order_id = int(line[0])
                item = line[2]  ## here item is a sku number
                name = line[3]
                # -------------------------------------------
                if order_id not in self.order_records: ## collect itemset for each order
                    self.order_records[order_id] = set({item})
                else:
                    self.order_records[order_id].add(item)
                # -------------------------------------------
                ## here we have to make sure not one sku maps to multi-names!
                if item not in self.sku_names:
                    self.sku_names[item] = name
                else:
                    if name != self.sku_names[item]:
                        print ('same sku has multi-names!', item, self.sku_names[item], name)
                        print ('error appears so stops!')
                        return
                # ---------------------------------------
                if item not in self.c_itemsets:  ## preare for k_items = 1, single item in each basket
                    self.c_itemsets.add((item,))


        self.transaction_times = len(self.order_records)

        print ('number of transactions= ', self.transaction_times)

        #for item in self.sku_names:
        #    if len(self.sku_names[item]) != 1: 
        #        print ('error', item, self.sku_names[item])
        #print ('ok')

        #return

        #print self.order_records
        ## ------- testing code : -----------------------------------------------
        #self.order_records = {1:set(['1','2','3','4']), 2:set(['1','2','4']), 3:set(['1','2']), \
        #               4:set(['2','3','4']), 5:set(['2','3']), 6:set(['3','4']), 7:set(['2','4'])}

        #self.c_itemsets = set({('1',),('2',),('3',),('4',)})
        #self.transaction_times = len(self.order_records)
        #self.sku_names = {'1':'Alice', '2':'Bob', '3': 'Carl','4':'David'}
        ## -----------------------------------------------------------------------

        self.prune(1)

        print ('1-item:', len(self.supports[1]))


        self.apriori_search()

        print (len(self.supports[1]), len(self.supports[2]))

        self.confidence = self.get_confidence()  ## only use when k_items = 2

# ---------------------------------------------------------

    ##  ATTENTION! this funsion is designed for k_itemset = 2, i.e. paired items

    def get_confidence(self):
        '''this method is used to compute confidence and lift for a two-item rule:
            confidence (A->B) = support(A and B)/support(B), just a conditional probability
            lift (A -> B) = support(A and B)/support(A)*support(B); A and B could be itemsets'''
        confidence ={}
        #for k in range(2,self.max_k+1):
        for k in range(2,3):
            itemsets = self.supports[k]
            for itemset in itemsets:
                for item in itemset:
                    confidence[(item,itemset)] = self.supports[k][itemset]/self.supports[1][(item,)] 


        #print (confidence)
        #return confidence

        with local_db:
            id =0
            cur = local_db.cursor()
            cur.execute("DELETE FROM beerhawk;")
            for item_itemset in confidence:
                id += 1
                item = item_itemset[0]
                basket = item_itemset[1]  ## basket is a tuple (item1, item2)
                #print basket
                support = self.supports[1][(item,)]

                if basket[0] == item: 
                    associate_item = basket[1]
                else:
                    associate_item = basket[0]

                #print (type(basket), type(item), type(basket[1]),type(self.sku_names[item]))

                lift = self.supports[2][basket]/self.supports[1][(basket[0],)]/self.supports[1][(basket[1],)]
                query = ("INSERT INTO beerhawk "
                         "(id, item, support, itemset, itemset_sup, confidence, lift) "
                         "VALUES (%(id)s,%(item)s,%(support)s,%(itemset)s,%(itemset_sup)s,%(confidence)s,%(lift)s);")
#                query_add = {'id': id, 'sku': item, 'support': support, 'itemset': str(item)+','+str(basket),'itemset_sup': self.supports[2][basket],'confidence':confidence[item_itemset], 'lift': lift}

                query_add = {'id':id, 'item': self.sku_names[item], 'support': support, 'itemset': self.sku_names[item]+'->'+self.sku_names[associate_item],'itemset_sup': self.supports[2][basket],'confidence':confidence[item_itemset], 'lift': lift}
                cur.execute(query, query_add)
            cur.fetchall()

        return confidence

# ---------------------------------------------------------

    def apriori_search(self):
        #self.itemsets[2] = {(1,64): 10, (1,2): 2, (2,64): 4}
        #self.itemsets[3] = {(2, 64, 191):1, (1, 2, 65):5, (2, 64, 96):80, (1, 2, 191):10}
        #self.c_itemsets = self.gen_itemsets(4)

        for k in range(2,self.max_k+1):
            self.c_itemsets = self.gen_itemsets(k)
            self.prune(k)


# ---------------------------------------------------------

    def prune(self, k_items):
        '''Prune itemsets whose support value < min_support'''
        #print 'start to prune?'
        #print self.c_itemsets

        #print '******************************'
        #print k_items
        ## first support is the integer frequency count for a proudct
        self.supports[k_items] = {}

        for order in self.order_records:
            order_set = self.order_records[order]
            if len(order_set) < k_items: continue  ## if # of items in itemsets < k_items, ignore the itemset
            #print order_set, '------'
            for itemset in self.c_itemsets:  ## here 'self.c_itemses' is the total possible
                #print 'itemset', itemset    ## combined k_items itemsets, {1,2}, {1,3}... it is a set.
                isSetExist = True
                for item in itemset:
                    isSetExist = item in order_set and isSetExist
                    ## to see an itemset like itemset = {1,2,3} in an order:{1,4,2,10,3..} = 'order_set'
                    ## check item=1,2,3 all exist in 'order_set'!
                    #print item, isSetExist#, order_set

                if isSetExist:
                    if itemset not in self.supports[k_items]:
                        self.supports[k_items][itemset] = 1
                    else:
                        self.supports[k_items][itemset] += 1


        #print 'raw', self.itemsets[k_items]

        self.c_itemsets = copy.copy(self.supports[k_items])

        ## note here I filter out the items by absolute supports(an interger), 
        ## not frequency/total frqeuency. Once after filtering, compute:
        ##    support = frequency/total frqeuency
        for itemset in self.c_itemsets:
            if self.c_itemsets[itemset] < self.min_support:
                del self.supports[k_items][itemset]
            else:
                self.supports[k_items][itemset] = float(self.supports[k_items][itemset])\
                    /self.transaction_times


# ---------------------------------------------------------

    def gen_itemsets(self, k_items):
        '''generate the possible combined itemsets. then later (prune method) we will 
           go to scan and check the fequency and the itemsets will bed tossed out
           if frequency (absolute support) < min_support'''

        #print k_items

        c_itemsets = set({})

        #print list(tuple(1))
        #print 'ok'

        ## k_item =1, c_itemsets = {1,2,3} = {(1,),(2,),(3,)}
        ## k_item =2, c_itemsets = {(1,),(2,),(3,)}*{(1,),(2,),(3,)} = {(1,2),(1,3),(2,3)}
        ## k_item =3, c_.. = {(1,2),(1,3),(2,3)}*{(1,),(2,),(3,)} = {(1,2,3)}
 
        ## {(1,),(2,),(3,)} are always the keys of 'self.supports[1]'
        ## {(1,2),(1,3),(2,3)} are keys of 'self.supports[2]'
        ## keys are always tuples sorted in numbers; set cannot store list, but key.
        ## key = (1,2), list(key) = [1,3] => [1,3,2] => [1,2,3] => (1,2,3) stored in c_itemsets.

        for itemset in self.supports[k_items-1]:
            ak = list(itemset)   ## even one-item, here itemset is a tuple, (item,)
                                 ## if itemset ='ABC', then list('ABC') = ['A','B','C']
                                 ## but list(('ABC',)) = ['ABC']
            #print ak, type(ak), itemset, type(itemset)
            #print '---', ak, '-----'
            for beer in self.supports[1]:
                ck = copy.copy(ak)
                if beer[0] not in ck: 
                    ck.append(beer[0])
                    ck.sort()
                    ck = tuple(ck)
                    #print ck#, len(ck), type(ck)
                    if len(ck) == k_items: c_itemsets.add(ck)
                
        #print len(c_itemsets)
        return c_itemsets




## arg: priori(min_support = int, max_k_items = [int>=2]) 

##  'min_support' is defined as the absolute support, so it is an integer.
##  but after filtering, the support will be divided by total trans so support becomes float

## ' max_k_items' is an integer needed >= 2. Meaning maximum number of items in an itemset.


A = Apriori_search(10, 2)


#print A.order_records
#print (A.supports[1])
#print ('')
#print (A.supports[2])
#print (A.supports[4])

