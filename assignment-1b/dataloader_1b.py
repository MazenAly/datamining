import numpy as np
from pandas import Series,DataFrame
import pandas as pd
from sklearn import metrics
from scipy import spatial
import dis
import math


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
    center0 =     points_list.sample(n=1, random_state=0, axis=0)
    centers_list = DataFrame(center0.drop(['distance' , 'center'] , axis = 1))
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
        print(centers_list)
        print("==============Cycle finished===========")
    centers_list.drop(centers_list.tail(1).index, inplace=True)
    print(points_list)
if __name__ == "__main__":
    c1 = load_data_1b("./data1b/C1.txt")
    c2 = load_data_1b("./data1b/C2.txt")
    c3 = load_data_1b("./data1b/C3.txt")
    gonzales(c3, 4)