from __future__ import print_function
import sys
import math
import numpy as np
import random
from operator import add

from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating

from pyspark import SparkContext, SparkConf

from pyspark.sql import SQLContext


def get_unpurchased_product(user, purchased_products, entire_products):
    '''randomly select unpurchased items for each user'''
    unpurchased_products = list(entire_products - set(purchased_products))
    return (user, random.sample(unpurchased_products, 20))
    
def Rank_list(product_nums):
    rank_list = {}
    for num in product_nums:
        if num == 1:
            rank_list[num] = [0.0]
        elif num == 2:
            rank_list[num] = [0.0, 100.0]
        elif num > 2:
            '''this convention considers rank percentage [0,x,x...,100%]'''
            increment = 100.0/(num-1)
            ranks = [0.0]
            for i in range(1, num-1):
               ranks.append(increment*i)

            ranks.append(100.0)
            rank_list[num]=ranks

    return rank_list

def model_rank_score(r, rank_lists):
    ''' 
     input: r = {user_a, [(beer_a1, rating_a1, prob_a1), (beer_a2, rating_a2, prob_a2), ....]}
                {user_b, [(beer_b1, rating_b1, prob_b1), (beer_b2, rating_b2, prob_b2), ....]} ...
     return:
              \sum_i rating_{u,i} * ranking_{u,i} for each user
    '''
    items = r[1]
    ## sort by probability in desc order; highest prob has early rank:
    items.sort(key=lambda tup: -tup[2])

    ''' 
    items[i][0] = i-th product, items[i][1] = i-th rating, items[i][2] = predicted probability for user r[0]
    '''

    if len(items) not in rank_lists:
        print (len(items), rank_lists)
        print (r[1])


    length = len(items)
    rating_ranking =0.0
    #sum_ratings = 0.0
    for i in range(length):
        rating_ranking += items[i][1]*rank_lists[length][i]
        #sum_ratings += items[i][1]

    #return (rating_ranking, sum_ratings)
    return rating_ranking



class ImplicitCF_engine(object):
    '''A collaborative filtering recommender engine using implicit feedback datasets:
       *spark.apache.org/docs/1.4.0/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS.trainImplicit
       *spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS.trainImplicit
       *spark.apache.org/docs/0.8.1/api/mllib/org/apache/spark/mllib/recommendation/ALS$.html
    '''
    
    def __init__(self, sc, sqlContext, masterIp):
        '''Init the recommendation engine given a Spark context and a dataset path'''

        self.sc = sc
        self.sqlContext = sqlContext
        self.masterIp = masterIp

        read_on = True

        if read_on:
            # self.users = sc.textFile("hdfs://"+masterIp+":9000/RDD/users").cache()
            # self.products = sc.textFile("hdfs://"+masterIp+":9000/RDD/products").cache()
            
            # self.user_product = sc.textFile("hdfs://"+masterIp+":9000/RDD/user_product")
            # self.user_product =self.user_product.map(lambda x: x[1:-1]).map(lambda x: x.split(",")).\
            #         map(lambda x: ((x[0][3:-1], x[1][3:-2]), int(x[2]))).\
            #         map(lambda x: Rating(int(x[0][0]), int(x[0][1]), float(x[1]))).cache()

            # self.num_user, self.num_product = self.users.count(), self.products.count()
            # print ('num of users:', self.num_user, 'num of products:', self.num_product)
            # self.training, self.test = self.partition_dataset(training_ratio=0.7)
            # self.training.cache(), self.test.cache()
            # return

            self.users = sc.textFile("hdfs://"+masterIp+":9000/RDD/users").cache()
            self.products = sc.textFile("hdfs://"+masterIp+":9000/RDD/products").cache()

            #print ('test:', self.users.take(10))
            #print ('training:', self.training.take(10))

            self.training = sc.textFile("hdfs://"+masterIp+":9000/RDD/training")
            self.test = sc.textFile("hdfs://"+masterIp+":9000/RDD/test")
            self.user_product = sc.textFile("hdfs://"+masterIp+":9000/RDD/user_product")

            # print ('test:', self.test.take(10))
            # print ('training:', self.training.take(10))

            self.training = self.training.map(lambda x: x[1:-1]).map(lambda x: x.split(", ")).\
                        map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2]))).cache()

            self.test =self.test.map(lambda x: x[1:-1]).map(lambda x: x.split(", ")).\
                        map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2]))).cache()

            self.user_product =self.user_product.map(lambda x: x[1:-1]).map(lambda x: x.split(", ")).\
                    map(lambda x: Rating(int(x[0]), int(x[1]), float(x[2]))).cache()
            
            #print ('test:', self.test.take(10))
            #print ('training:', self.training.take(10))
            # exit()

            self.num_user, self.num_product = self.users.count(), self.products.count()
            print ('num of users:', self.num_user, 'num of products:', self.num_product)
            print ('total (user, product) pair', self.user_product.count())
            return

    
        self.get_data(sc, sqlContext)

        self.users, self.products = self.get_user_product_RDD()
        self.users.cache(), self.products.cache()

        self.training, self.test = self.partition_dataset(training_ratio=0.7)
        self.training.cache(), self.test.cache()

        print ('before adjustment, training:', self.training.count(), ', test:', self.test.count())

        self.adjust_dataset()
        self.assign_unpurchased_testdata()

        #self.saveRDD()
        

    def get_data(self, sc, sqlContext):
        '''
        reading data from HDFS and prpeare (user, product) RDD
        '''
        orders = sc.textFile(sys.argv[1], 1)#.zipWithIndex().filter(lambda x: x[1] < 342108).keys()
        # prepare SQL df |'order_id'|'user_id'| from 'orders.csv' (select only eval_set = 'prior')
        header = orders.take(1)[0] # find the header and remove later
        orders_RDD = orders.filter(lambda x: x!= header).map(lambda x: x.split(",")).filter(lambda x : x[2] == 'prior').map(lambda x: (x[0],x[1]))
        order_user_DF = sqlContext.createDataFrame(orders_RDD, ["order_id","user_id"])
        order_user_DF.registerTempTable("orders_users") # assign SQL df |'order_id'|'user_id'|

        # ----------------------
        orders_prior = sc.textFile("hdfs://"+self.masterIp+":9000/user/order_products__prior.csv")#.zipWithIndex().filter(lambda x: x[1] < 3243449).keys()
        # prepare SQL df |'order_id'|'product_id'| from 'order_product__prior.csv'
        header = orders_prior.take(1)[0] # find the header and remove later
        orders_prior_RDD = orders_prior.filter(lambda x: x!= header).map(lambda x: x.split(",")).map(lambda x: (x[0],x[1]))
        order_product_DF = sqlContext.createDataFrame(orders_prior_RDD, ["order_id","product_id"]) # assign SQL df |'order_id'|'product_id'|
        order_product_DF.registerTempTable("orders_products")

        # join DF: 'order_user_DataFrame' and 'order_product_DataFrame' to get |'order_id'|'user_id'|'product_id'|
        user_product_DataFrame = sqlContext.sql('''select orders_users.order_id, user_id, product_id \
                       from orders_users join orders_products on (orders_users.order_id=orders_products.order_id)''')
        #user_product_DataFrame.show(10)

        self.user_product = user_product_DataFrame.rdd.map(lambda x: ((x['user_id'], x['product_id']),1)).reduceByKey(add).\
                    map(lambda x: Rating(int(x[0][0]), int(x[0][1]), float(x[1]))).cache()



    def assign_unpurchased_testdata(self):
        '''for each user in test dataset, add x more unpurchased products'''
        purchased_products_RDD = self.test.map(lambda x: (x.user, [x.product])).reduceByKey(add).cache()

        products_set = set(self.products.collect())

        unpurchased_products_RDD = purchased_products_RDD.map(lambda x: get_unpurchased_product(x[0], x[1], products_set)).\
                        flatMap(lambda x: [Rating(int(x[0]), int(v), 0.001) for v in x[1]]).cache()
        #print (unpurchased_products_RDD.filter(lambda x: x[0] in set({r'92396', r'198196', r'111182'})).collect())
        self.test = self.test.union(unpurchased_products_RDD).cache()


        
    def adjust_dataset(self):
        '''to check if each user and each product appears in training/test datasets and
           to redistribute training/test datasets
        '''
        
        training_RDD = self.training.map(lambda x: (x.user, 1)).reduceByKey(add).cache()

        self.sqlContext.createDataFrame(self.users.map(lambda x: (x, 1)), ["user", "freq"]).registerTempTable("users")
        self.sqlContext.createDataFrame(training_RDD, ["user", "freq"]).registerTempTable("trainingUsers")
        self.sqlContext.createDataFrame(self.test.map(lambda x: (x.user, x.product, x.rating)), ["user", "product", "freq"]).registerTempTable("test_userproduct")
        
        
        self.sqlContext.sql('''select distinct a.user \
                               from users a left join trainingUsers b on (a.user=b.user) \
                               where b.user is null''').registerTempTable("missing_users_training")
        #missing_users_intraining_DF.show(5)
        #missing_users_intraining = set(missing_users_intraining_DF.rdd.map(lambda x: x['user']).collect()) # the users that training dateset doesn't have (compared to the whole user list)
        #print (len(missing_users_intraining), missing_users_intraining)
        
        fetch_users_fromtest_DF = self.sqlContext.sql('''select a.user, MAX(b.product) as product \
                                                   from missing_users_training a join test_userproduct b on (a.user=b.user) \
                                                   group by a.user \
                                                   order by a.user''').cache()

        #fetch_users_fromtest_DF.show(30)

        fetch_users_fromtest = set(fetch_users_fromtest_DF.rdd.map(lambda x: (x['user'], x['product'])).collect())
        #print (len(fetch_users_fromtest))

        users_fromtest_RDD = self.test.filter(lambda x: (x.user, x.product) in fetch_users_fromtest).cache()

        self.training = self.training.union(users_fromtest_RDD).cache()
        self.test = self.test.filter(lambda x: (x.user, x.product) not in fetch_users_fromtest).cache()

        

        training_RDD = self.training.map(lambda x: (x.product, 1)).reduceByKey(add).cache()

        self.sqlContext.createDataFrame(self.products.map(lambda x: (x, 1)), ["product", "freq"]).registerTempTable("products")
        self.sqlContext.createDataFrame(training_RDD, ["product", "freq"]).registerTempTable("trainingProducts")
    
        self.sqlContext.sql('''select distinct a.product \
                                                   from products a left join trainingProducts b on (a.product=b.product) \
                                                   where b.product is null''').registerTempTable("missing_products_training")

        fetch_products_fromtest_DF = self.sqlContext.sql('''select MAX(b.user) as user, a.product \
                                                   from missing_products_training a join test_userproduct b on (a.product=b.product) \
                                                   group by a.product \
                                                   order by a.product''').cache()
        #fetch_products_fromtest_DF.show(10)

        fetch_products_fromtest = set(fetch_products_fromtest_DF.rdd.map(lambda x: (x['user'], x['product'])).collect())

        #print (len(fetch_products_fromtest))

        products_fromtest_RDD = self.test.filter(lambda x: (x.user, x.product) in fetch_products_fromtest).cache()

        self.training = self.training.union(products_fromtest_RDD).cache()
        self.test = self.test.filter(lambda x: (x.user, x.product) not in fetch_products_fromtest).cache()
        


    def saveRDD(self):
        '''to save RDD'''
        self.users.saveAsTextFile ("hdfs://"+self.masterIp+":9000/RDD/users")
        self.products.saveAsTextFile ("hdfs://"+self.masterIp+":9000/RDD/products")

        self.training.map(lambda x: (x.user, x.product, x.rating)).saveAsTextFile ("hdfs://"+self.masterIp+":9000/RDD/training")
        self.test.map(lambda x: (x.user, x.product, x.rating)).saveAsTextFile ("hdfs://"+self.masterIp+":9000/RDD/test")
        self.user_product.map(lambda x: (x.user, x.product, x.rating)).saveAsTextFile ("hdfs://"+self.masterIp+":9000/RDD/user_product")



    def get_user_product_RDD(self):
        '''This method is used to get the user RDD and product RDD
           input: (user, product) RDD
           output: users RDD (John, Alex, Bob, Ken...) and products RDD (item1, item2,..)
        '''
        users_RDD = self.user_product.map(lambda x: (x.user, 1)).reduceByKey(add).map(lambda x: x[0])
        products_RDD = self.user_product.map(lambda x: (x.product, 1)).reduceByKey(add).map(lambda x: x[0])

        self.num_user, self.num_product = users_RDD.count(), products_RDD.count()
        print ('number of users:', self.num_user, ', number of products:', self.num_product)

        return users_RDD, products_RDD



    def partition_dataset(self, training_ratio=0.7):
        '''To partition entire dataset to 70% training and 30% test datasets; also make sure
           the (user,product) pair which appears more than once will not be in test dataset
           (1) assign all (user,product) pairs which appear more than one time to training_1 dataset
           (2) then partition the remaining dataset to traning_2 and test datasets, such that training_1 + training_2
               is about 70% and the test datadate is about 30%.

           input: whole (user, product) dataset
           output: training and test datasets (default by 70%/30% rule)
        '''
        
        ## assign (user_id, product_id) at-least-once appearance to training set
        training_RDD = self.user_product.filter(lambda x: x.rating > 1)

        num_user_product = self.user_product.count()
        num_training_needed = int(num_user_product*training_ratio)
        num_atleast_once_user_product = training_RDD.count()

        print ('total pair:', num_user_product)
        #print ('70% should be:', num_training_needed)
        #print ('>1 pair:', num_atleast_once_user_product)

        fraction = float(num_training_needed - num_atleast_once_user_product) / (num_user_product - num_atleast_once_user_product)

        # the RDD that (user,product) appears exactly one time only, and use this to partition training_2/test datasets
        # if the previous training_1 is about 30%, then we just need another 40% for training_2 to make 70% training dataset and the remaining 30% for test dataset.
        once_user_product_RDD = self.user_product.filter(lambda x: x.rating == 1)

        training_2_RDD, test_RDD = once_user_product_RDD.randomSplit([fraction, 1.0-fraction], seed=np.random.randint(1,9999))
        # training_2 will be union with the self.training which previously is formed from (user,product) appear more than one times such that self.training + triaining_2 is about 70% of the overall data.
        training_RDD = training_RDD.union(training_2_RDD)

        return training_RDD, test_RDD



    def training_models(self, rank=5, seed=32, iterations=20, alpha=0.01, reg=0.01):
        '''ALS training parameters:
            rank - Number of latent factors.
            iterations - Number of iterations of ALS. (default: 5)
            lambda_ - Regularization parameter. (default: 0.01)
            alpha - constant used in computing confidence. (default: 0.01)
            seed - Random seed for initial matrix factorization model. (default: None)
        '''

        print (self.training.take(5), self.test.take(5))

        weights = [.8, .2]
        trainData_RDD, valData_RDD = self.training.randomSplit(weights, seed)  # split training to training and validation sets

        trainData_RDD.cache(), valData_RDD.cache()

        print (trainData_RDD.count(), valData_RDD.count())


        #X_val_RDD = valData_RDD.map(lambda x: (x.user, x.product)).filter(lambda x: x[0] in set({92396, 198196, 111182, 2350, 46158})).cache()
        X_val_RDD = valData_RDD.map(lambda x: (x.user, x.product)).cache()
   
        sum_ratings_val = valData_RDD.map(lambda x: x.rating).sum()

        product_nums_for_users = X_val_RDD.map(lambda x: (x[0], 1)).reduceByKey(add).map(lambda x: x[1]).collect()
        #print (X_val_RDD.collect())
        print ('num of users', X_val_RDD.map(lambda x: (x[0], 1)).reduceByKey(add).count())
        #print (product_num_for_users)
        rank_lists = Rank_list(product_nums_for_users)

        print (rank_lists)
        #print (rank_lists[4])

        #return

        model = ALS.trainImplicit(trainData_RDD, rank, iterations=iterations,\
                            lambda_=reg, blocks=-1, alpha=alpha, nonnegative=False, seed=seed)

        # prediced results for validation results
        predictions_RDD = model.predictAll(X_val_RDD).map(lambda x: ((x[0], x[1]), x[2]))
        ratings_and_preds_RDD = valData_RDD.map(lambda x: ((x[0], x[1]), x[2])).join(predictions_RDD)

        print()
        print('model training is convergenent')
        print()
        #return

        MPR = self.percentage_ranking(ratings_and_preds_RDD, rank_lists, sum_ratings_val)


        print ('Rank %s, reg %s, alpha %s, AvgRank = %s' % (rank, reg, alpha, MPR))


    def percentage_ranking(self, ratings_and_predictions, rank_lists, sum_ratings):
        ''' This function is sued to compute average ranking: mean percentage ranking (MPR).
            input: 
                'ratings_and predictions' is a RDD of 
                ((uer1, product1),(rating1, prob1)), ((uer2, product2),(rating2, prob2))....
                ((user=x[0][0],product=x[0][1]),(rating=x[1][0],prob=x[1][1]))
                'rank_lists' = [0%, xxxx.... 100%] ranking in terms of %.
            return:
                avg model_rating_rank = \sum_{u,i} r_{u,i}* rank_{u,i} / \sum_{u,i} r_{u,i}
        '''

        users_predictions_RDD = ratings_and_predictions.map(lambda x: (x[0][0], [(x[0][1], x[1][0], x[1][1])])).reduceByKey(add)
        '''
            now 'users_predictions' is a RDD of
                    {user1, [(product11, rating11, prob11), (product12, rating12, prob12), ...]},
                    {user2, [(product21, rating21, prob21), (product22, rating22, prob22).....]},...
                    {user, [(beers, ratings, probs)]}....
        '''

        print ('user_predictions is ok')

        sum_ratings_rankings = users_predictions_RDD.map(lambda x: model_rank_score(x, rank_lists)).sum()
        ''' 'users_prediction.map(lambda x: model_rank_score..)' gives a RDD of 
            \sum_i r_{ui}* rank_{ui} for each user u.
            after .sum(), it becomes to \sum_{u,i} r_{ui}*rank_{ui} 
        '''

        return sum_ratings_rankings/sum_ratings


        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sort <file>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="orders")
    sqlContext = SQLContext(sc)
    master_ip = "54.82.188.27"

    #rdd = sc.parallelize([1, 2, 3, 4, 5, 6])
    #aset = set({2,3,5})
    #print (rdd.map(lambda x: x%2 != 0).collect())
    #print (rdd.filter(lambda x: x not in aset).collect())
    
    # rdd = sc.parallelize([(('a', 4), 1), (('a',4), 2), (('b',1),3),(('a',0), 4), (('c',0),5),(('c',1), 6)])
    # aset = set({'a'})
    # bset = set({4,1})
    # print (rdd.filter(lambda x: x[0][0] not in aset).collect())
    # print (rdd.filter(lambda x: x[0][1] not in bset).filter(lambda x: x[0][0] not in aset).collect())

    # rdd = sc.parallelize([('a', [41]), ('a',[4]), ('b',[13]),('a',[0]), ('c',[5]),('c',[16]) ])
    # rdd = rdd.reduceByKey(add)
    # print (rdd.reduceByKey(add).collect())
    # rdd = rdd.map(lambda x: get_unpurchased_product(x[0], x[1], set({41,4,13,0,5,16})))

    # print (rdd.collect())

    # exit()

    
    engine = ImplicitCF_engine(sc, sqlContext, master_ip)
    
    print ('after adjustment, training:', engine.training.count(), ', test:', engine.test.count())
    #print (engine.training.take(5), engine.test.take(5))

    engine.training_models()

    

    sc.stop()
