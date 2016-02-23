import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import  sklearn.cluster 
from scipy import spatial
import dis
import math
import matplotlib.pyplot as plt
from random import randint

def load_data_1b(fpath):
    data = []
    f = open(fpath, 'r')
    for line in f:
        words = line.split()
        data.append(words)
    f.close()
    arr = np.array(data, dtype=np.float64)
    return arr

def gonzales(data , k):
    #transform the data numpy array to data frame using the id as index
    points_list = DataFrame(data[:, 1:] , index = data[ : , 0])
    #adding two columns in the points data frame for saving the centers and distance
    points_list["distance"] = np.nan
    points_list["center"] = np.nan
    distance_column_index = points_list.columns.get_loc("distance")
    #choosing a random point as the first center

    center0 =     points_list.sample(n=1 , random_state = randint(0,100) , axis=0)
    centers_list = DataFrame(center0.drop(['distance' , 'center'] , axis = 1))
    centers_list['color'] = 'r'
    colors = "bgcmykw"
    print(centers_list)
    print("==============Initialization finished===========")
    #looping k-1 time to have k centers
    for k_cycle in range(1,k+1):
        # varibles to save the next center to be chosen based on the maximum distance a point makes within its cluster
        max_distance = 0 
        next_cluster = np.nan
        #loop on all the points to assign them to their closest center 
        for indexp, p in points_list.iterrows():
            #variables to save the choose the closest center
            min_cluster_distance = math.inf
            clostest_cluster = None
            for indexc, center in centers_list.iterrows():
                dis = spatial.distance.euclidean(center.as_matrix(columns=[0 ,1]) , p.as_matrix(columns=[0 ,1]))
                if dis < min_cluster_distance:
                    min_cluster_distance = dis
                    clostest_cluster = indexc
            p["distance"] = min_cluster_distance
            p["center"] = clostest_cluster               
            if min_cluster_distance > max_distance:
                max_distance = min_cluster_distance
                next_cluster = indexp 
            
        centers_list = centers_list.append(points_list.ix[[next_cluster], :distance_column_index   ])
        centers_list.set_value(next_cluster, 'color', colors[k_cycle])
        print(centers_list)
        print("==============Cycle finished===========")
    centers_list.drop(centers_list.tail(1).index, inplace=True)
    centers_list.drop(['color'], axis=1 ,inplace=True)


    centers_list.plot(kind='scatter', x=0, y=1 , c='r'   )
    points_list.plot(kind='scatter', x=0, y=1 , c='center' , s= points_list['center'] *2   )
    plt.show()

    print(points_list)
    return centers_list.as_matrix(columns=[0 ,1])

def kmeans_scikit(data , k):
    points_list = DataFrame(data[:, 1:] , index = data[ : , 0])
    mat = points_list.as_matrix()
    print(mat)
    # Using sklearn
    km = sklearn.cluster.KMeans(n_clusters=k)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    print(labels)
    print('==============')
    print(km.predict([[20 ,-15]]))
    # Format results as a DataFrame
    #results = pd.DataFrame([points_list.index,labels]).T
    points_list['labels'] = labels
    points_list.plot(kind='scatter', x=0, y=1    , c='labels'  )
    plt.show()
    print(points_list)

if __name__ == "__main__":
    c1 = load_data_1b("./data1b/C1.txt")
    c2 = load_data_1b("./data1b/C2.txt")
    c3 = load_data_1b("./data1b/C3.txt")
    #kmeans_scikit(c1 , 4)
    gonzales(c2, 3)
