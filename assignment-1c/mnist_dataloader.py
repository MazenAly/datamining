# Source: http://neuralnetworksanddeeplearning.com/chap1.html
# You need only train and test sets. Simply disregard the validation set.
import numpy as np
from math import sqrt
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
import time 
import math
import sys
import matplotlib.pyplot as plt

# function that takes two vectors and return the dot product value of it by summing the multiplication of each two elements 
#def dot_product(v1, v2):
#    return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

def cosine_similarity(v1, v2):
    prod = np.dot(v1, v2)
    # len and len2 are the magnitudes of the 2 vectors
    len1 = np.sqrt(np.dot(v1, v1))
    len2 = np.sqrt(np.dot(v2, v2))
    return np.divide(prod , np.multiply(len1 , len2))


def euclidean_distance(v1, v2):
    diff = np.subtract(v1, v2)
    return np.linalg.norm(diff)


def load_data():
    f = gzip.open('./data1a/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f , encoding='latin1' )  #added encoding for Python 3
    f.close()
    return training_data, validation_data, test_data

def create_matrix(d, k): 
    
    mtx_val = 1/np.sqrt(d)
    prob = 0
    
    mtx = np.zeros((d, k))
    for i in range(0,d):
        for j in range(0,k):
            random_number = np.random.random_sample()
            if random_number <= 0.5:
                mtx[i,j] = mtx_val
            else:
                mtx[i,j] = -mtx_val
            
    return mtx
    #------------------------------------------ for i in range(len(data_clone)):
    #------------------ prob = prob + np.divide(distances[i], np.sum(distances))
    #------------------------------------------------- if random_number <= prob:

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


print("start")
training_data, validation_data, test_data  = load_data_wrapper()
training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data) # for transforming the zip objects to lists

training_data = [[entry[0].flatten(), entry[1]] for entry in training_data]
test_data = [[entry[0].flatten(), entry[1]] for entry in test_data]

print('training data', len(training_data[0][1]))

training_data_content = np.matrix([training_data[i][0] for i in range(len(training_data))])
test_data_content = np.matrix([test_data[i][0] for i in range(len(test_data))]) 

k = int(sys.argv[1])
d = len(training_data[0][0])
mtx = np.matrix(create_matrix(d, k))
        
training_data_content = training_data_content * mtx
test_data_content = test_data_content * mtx  
training_data_content = np.multiply(training_data_content, np.sqrt(np.divide(d, k)))
test_data_content = np.multiply(test_data_content, np.sqrt(np.divide(d, k)))

training_data_content = [np.array(training_data_content[i]).flatten() for i in range(len(training_data_content))]
test_data_content = [np.array(test_data_content[i]).flatten() for i in range(len(test_data_content))]

k_values = [50 , 100 , 500]

#for k_value in k_values:
#    mtx = np.matrix(create_matrix(784, k_value))
#    training_data_projected = training_data * mtx
#    test_data_projected = test_data * mtx  
#    
#    print('training data proj', training_data_projected.shape)
#    print('test data proj', test_data_projected.shape)
#    
#    distored_values_list = []
#    for i in range(20):
#        for j in range(i+1, 20):
#            dis_orig = spatial.distance.euclidean(training_data[i,:] , training_data[j,:])
#            dis_projected = spatial.distance.euclidean(training_data_projected[i,:] , training_data_projected[j,:])
#            distored_values_list.append(dis_projected  /dis_orig )
#    
#    #--------------------------- for i , val in enumerate(distored_values_list):
#        #---------------------------------------------------- plt.scatter(i,val)
#        #--------------------------------------------------- plt.xlabel("pairs")
#        #--------------------------------------- plt.ylabel("Distortion value ")
#    #------------------------------------------------------------------ plt.show
#    
#    print(len(distored_values_list))
#    plt.hist(distored_values_list)
#    axes = plt.gca()
#    #axes.set_xlim([xmin,xmax])
#    axes.set_ylim([0,70])
#    plt.title("Distortion Histogram for k = " + str(k_value))
#    plt.xlabel("Distortion value for k = " + str(k_value))
#    plt.ylabel("Frequency")
#    plt.show()



#print(test_data)


#module for digits classification using nearest neighbor classifier
def NNClassifier():
    #opening two files for writing the predications and the true labels.
    f_predicted = open('predicted_' + str(k) + '.txt', 'w')
    f_label = open('label_' + str(k) + '.txt', 'w')
    #for each test entry, we will loop on every training entry and check if 'ts the closest neighbor so far or not.
    for i in range(1000):
        test_entry = test_data[i]
        start_time = time.time()
        #test_features = test_entry[0]    
        test_features = test_data_content[i]
        test_label = test_entry[1]
        # defining variables to save the closest neighbor
        min_dis = math.inf
        nearest_neighbor = 0 
        for j in range(len(training_data)):
            training_entry = training_data[j]
            #training_entry_features = training_entry[0]
            training_entry_features = training_data_content[j]
            #measuring the cosine similarity using scipy library
            #sim = 1 - spatial.distance.cosine(training_entry_features, test_features)
            #sim = cosine_similarity(training_entry_features, test_features)
            distance = euclidean_distance(training_entry_features, test_features)
            #distance = np.sum(np.power(np.subtract(training_entry_features, test_features), 2))
   
            if distance < min_dis: # if this entry has the max similarity so far then safe this entry as nearest neighbor
                min_dis = distance 
                nearest_neighbor = training_entry
        # to know the predicated label we get the index of the 1 value, as the label of the training data are an array of 10 values, zeros for all and 1 for the label      
        predicated_label = list(nearest_neighbor[1]).index(1)
        print(predicated_label)
        print(test_label)
        print(nearest_neighbor[1][test_label])
        #writing the predicated and the actual labels in the files
        f_predicted.write(str(predicated_label) + '\n');
        f_label.write(str(test_label) + '\n');
        #check if the they are the same, but as the training label is array and test label is just the number, we are using the condition below
        if nearest_neighbor[1][test_label] == 1:
            print("success")
        else:
            print("fail")
        print("=============")
        print('duration', time.time() - start_time)
    # closing the files
    f_predicted.close()
    f_label.close()

# defining the confusion matrix
mtx = np.zeros((11, 11))
#printing options of numpy
np.set_printoptions(suppress=True) 
#function to build the confusion matrix, rows are the predicted labels and columns are true labels
def build_confusion_matrix():
    f_predicted = open('predicted_' + str(k) + '.txt', 'r')
    f_label = open('label_' + str(k) + '.txt', 'r')
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

# function that takes the confusion matrix as input and measure the accuracy of the classifier, as well as the precision and recall for each class  
def print_analytics(mtx):  
    # the accuracy is measured by dividing the number of correct predictions by the total number of predictions
    print("Accuracy of the NN classifier is : " , sum(mtx.diagonal()[:10]) * 100/1000, "%"  ) 
    for label in range(0,10): # for each label we measure the precision and recall
        print("Class " , label , ":")
        # the precision is measured by dividing the true positives by the summation of true positives and false positives of this class
        print("Precision:" , mtx[label][label] / mtx[label][10])
        # the recall is measured by dividing the true positives by the summation of true positives and false negatives of this class  
        print("Recall:" , mtx[label][label] / mtx[10][label] )  
    
# using scikit learn to do k-neighbor classification
def nn_scikit_learn():
    print("===========Scikit learn Knn=======================")
    training_data, validation_data, test_data  = load_data()
    training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)
    start_time = time.time() 
    # defining Scikit learn K neighbors classifier with metric cosine, and number of neighbors to check as just 1 with algorithm brute 
    #instead of the default 'auto' as this option works with the cosine metric
    knn = KNeighborsClassifier(n_neighbors = 1 , metric = 'cosine' , algorithm='brute' )
    # fit function takes the training feature as the first argument and training labels as second argument
    knn.fit(training_data[0] , training_data[1])
    # getting the predicted labels for the test data.
    Y_pred = knn.predict(test_data[0])
    #measuring the accuracy
    print(metrics.accuracy_score(test_data[1] ,Y_pred))
    end_time = time.time()
    print("elapsed time: " , end_time - start_time)

#--------------------------------------------------------- if len(sys.argv) > 1:
    #--------------------------------------------------------- nn_scikit_learn()
#------------------------------------------------------------------------- else:
NNClassifier()

#confusion_matrix = build_confusion_matrix()
#print_analytics(confusion_matrix)

