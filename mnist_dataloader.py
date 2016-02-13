# Source: http://neuralnetworksanddeeplearning.com/chap1.html
# You need only train and test sets. Simply disregard the validation set.
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle
import gzip
import pprint
pp = pprint.PrettyPrinter(indent=4)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from scipy import spatial

import math

def dot_product(v1, v2):
    return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

def cosine_similarity(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)

def load_data():
    f = gzip.open('./data1a/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f , encoding='latin1' )
    f.close()
    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


training_data, validation_data, test_data  = load_data_wrapper()
training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)


def NNmodule():
    f_predicted = open('predicted.txt', 'w')
    f_label = open('label.txt', 'w')
    for test_entry in test_data:
        test_features = test_entry[0]
        test_label = test_entry[1]
        max_sim = 0  
        nearest_neighbor = 0 
        for training_entry in training_data:
            training_entry_features = training_entry[0]
            sim = 1 - spatial.distance.cosine(training_entry_features, test_features)
            if sim > max_sim: 
                max_sim = sim 
                nearest_neighbor = training_entry
        predicated_label = list(nearest_neighbor[1]).index(1)
        print(predicated_label)
        print(test_label)
        print(nearest_neighbor[1][test_label])

        f_predicted.write(str(predicated_label) + '\n');
        f_label.write(str(test_label) + '\n');

        if nearest_neighbor[1][test_label] == 1:
            print("success")
        else:
            print("fail")
        print("=============")

    f_predicted.close()
    f_label.close()


mtx = np.zeros((11, 11))
np.set_printoptions(suppress=True)
def build_matrix():
    f_predicted = open('predicted.txt', 'r')
    f_label = open('label.txt', 'r')
    for predicted_label in f_predicted:
        true_label = f_label.readline()
        mtx[int(predicted_label)][int(true_label)] += 1
        print(mtx) 
        print("====")
    for i in range(0,10):
        mtx[i][10] = sum(mtx[i,:10])
        mtx[10][i] = sum(mtx[:10, i])  
    mtx[10][10] =   sum(mtx[10,:10]) + sum(mtx[:10,10])
    print("Final confusion matrix:")
    print(mtx)
    return mtx    
    
def print_analytics(mtx):  
    print("Accuracy of the system is : " , sum(mtx.diagonal()[:10]) * 100/len(test_data) , "%"  ) 
    for label in range(0,10):
        print("Class " , label , ":")
        print("Precision:" , mtx[label][label] / mtx[label][10])
        print("Recall:" , mtx[label][label] / mtx[10][label] )  
    
#NNmodule()   
confusion_matrix = build_matrix()
print_analytics(confusion_matrix)





#===============================================================================
# mtx = np.array([[  978.  ,   0   ,  8.  ,  0   ,  0  ,   1.  ,   3.   ,  1.,     4.   ,  9.,
#       0] ,
#  [    1. , 1128.  ,   0    , 0   ,  3.  ,   0  ,   3. ,  11.    , 2.   ,  6.,
#       0],
#  [    0   ,  3. , 1005.  ,   1.  ,   1.  ,   0   ,  0 ,    5.   ,  2. ,    1.,
#       0],
#  [    0  ,   1.   ,  5. ,  974.  ,   0  ,  19.   ,  0   ,  2. ,   15.  ,   4.,
#       0],
#  [    0   ,  1.  ,   0   ,  1.   ,937.   ,  1.   ,  2.  ,   1. ,    2. ,    9.,
#       0],
#  [    0   ,  1.   ,  0   , 14. ,   0 ,  847.    , 3.    , 0 ,    4.  ,   2.,
#       0],
#  [    0   ,  1.  ,   1.    , 0   ,  6. ,   11. ,  947.   ,  0  ,   5.   ,  1.,
#       0],
#  [    1.   ,  0   , 10  ,   4.   ,  2. ,    1.   ,  0 , 997.  ,   5.   ,  9.,
#       0],
#  [    0   ,  0    , 2.  ,   8.   ,  1.    , 6.   ,  0  ,   0 ,  931.  ,   4.,
#       0],
#  [    0  ,   0  ,   1.  ,   8.   , 32.  ,   6.  ,   0 ,   11.  ,   4. ,  964.,
#       0],
#  [    0   ,  0    , 0  ,   0    , 0   ,  0  ,   0   ,  0   ,  0    , 0,
#       0]])
#===============================================================================

 

#----------------------------------------------------- for row in training_data:
    #----------------------------------------------------------------- x = x + 1
    #------------------------------------------------------------------ print(x)
    #----------------------------------------- #sim = cosine_similarity(row[0], yy)
    #----------------------------- sim = 1 - spatial.distance.cosine(row[0], yy)
    #--------------------------------------------------------- if sim > max_sim:
        #--------------------------------------------------------- max_sim = sim
        #------------------------------------------------------- result = row[1]
#------------------------------------------------------------------------------ 
#------------------- print(result[test_data[0][1]] ,test_data[0][1] , max_sim  )
#-------------------------------------------------------- print("=============")
    

#------------------------------------------------------ dataSetI = [3, 45, 7, 2]
#--------------------------------------------------- dataSetII = [2, 54, 13, 15]
#--------------------- result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

#------------------- print("===========Scikit learn Knn=======================")
#------------------------------------------------------------------------------ 
#---------------------- training_data, validation_data, test_data  = load_data()
# training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)
#------------------------------------------------------------------------------ 
# knn = KNeighborsClassifier(n_neighbors = 1 , metric = 'cosine' , algorithm='brute' )
#---------------------------------- knn.fit(training_data[0] , training_data[1])
#-------------------------------------------- Y_pred = knn.predict(test_data[0])
#--------------------------- print(metrics.accuracy_score(test_data[1] ,Y_pred))
