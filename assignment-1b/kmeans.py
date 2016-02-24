import numpy as np
from plotting import *
from scipy import spatial
import math
import sys
import matplotlib.pyplot as plt
MAX_NUMBER = math.inf 
import math
import gonzales as mn 

def load_data_1b(fpath):
    data = []
    f = open(fpath, 'r')
    for line in f:
        words = line.split()
        data.append(words)
    f.close()
    arr = np.array(data, dtype=np.float64)
    return arr[:, 0:]

def cluster_distance(cluster1, cluster2):
    center1 = np.average(cluster1,0)
    center2 = np.average(cluster2,0)

    if np.isnan(center1).any():
        center1 = [0,0]
    if np.isnan(center2).any():
        center2 = [0,0]

    return np.linalg.norm(np.array(center1) - np.array(center2))

def choose_k_center(data, k):
    data_clone = np.copy(data)
    seeds = [data[np.random.random_integers(len(data) - 1)]]
    for r in range(1,k):
        distances = []
        for point in data_clone:
            min_dist = MAX_NUMBER
            for i in range(r):
                dist = np.linalg.norm(point - seeds[i])
                if dist < min_dist:
                    min_dist = dist
            
            distances.append(np.power(min_dist, 2))

        random_number = np.random.random_sample()
        prob = 0
        for i in range(len(data_clone)):
            prob = prob + np.divide(distances[i], np.sum(distances))
            if random_number <= prob:
                seeds.append(data_clone[i])
                data_clone = np.delete(data_clone, i, 0)
                break;

    return seeds

def kmeans(data, k, method):
    original_data = data 
    data = data[: , 1:]
    if method == 'firstk':
        seeding_points = data[0:k]
    elif method == 'random':
        random_indices = np.random.random_integers(len(data) - 1, size = (1,k)).flatten()
        seeding_points = [data[i] for i in random_indices]
    elif method == 'kmeans++':
        seeding_points = choose_k_center(data, k)
    else:
        seeding_points = mn.gonzales(original_data , k)

    old_clusters = [[]]
    new_clusters = [[] for i in range(k)]
    
    costs_array = []
    while True:
        #=======================================================================
        # print('process next round')
        # print('seeding_points', seeding_points)
        #=======================================================================
        old_clusters = new_clusters
        new_clusters = [[] for i in range(k)]
        cost = 0
        for point in data:
            min_dist = MAX_NUMBER
            current_index = 0
            for i in range(k):
                seed = seeding_points[i]
                dist = np.linalg.norm(point - seed)
                if dist < min_dist:
                    min_dist = dist
                    current_index = i
            new_clusters[current_index].append(point)
            cost += np.power(np.linalg.norm(point - seeding_points[current_index]), 2)
         
        #print('==============')
        seeding_points = [np.average(new_clusters[i],0) for i in range(k)]
        distance = np.sum([cluster_distance(old_clusters[i], new_clusters[i]) for i in range(k)])
        #print('cost', cost)
        costs_array.append(cost)
        if distance == 0:
            break

    return new_clusters, cost , costs_array



def run_analysis():
    print("analytics")
    clusters_no = [3,4,5]
    data = load_data_1b("./data1b/C2.txt")
    methods = ['firstk' ,'random' ,'kmeans++' ,'gonzales']
        
    for m in methods:
        print("method ================================================" , m)
        k_costs= []
        k_std= []
        for k in clusters_no:
            print("clusters " , k)
            all_scores_k = []
            plt.figure(1)  
            longest_run = 0
            for i in range(0,5):
                result , cost , costs_array = kmeans(data, k, m)
                plt.plot(range(0,len(costs_array)), costs_array  , label = 'run '+ str(i) )
                plt.legend(loc='upper right', shadow=True) 
                plt.xlabel('Steps by ' + str(m) + ' method for ' + str(k) +' clusters')
                if longest_run < len(costs_array):
                    longest_run = len(costs_array)
                    plt.xticks(range(0,longest_run))
                
                plt.ylabel('Cost')
                #scatterplot(result)
                all_scores_k.append(cost)
            plt.show()
            avg_cost=np.array(all_scores_k).mean()
            print(*all_scores_k)
            std = np.std(np.array(all_scores_k))
            k_costs.append(avg_cost)
            k_std.append(std)
            print("Avg cost for " , m , 'and clusters no ' , k , " is " , avg_cost)
            print("Standard Deviation " , m , 'and clusters no ' , k , " is " , std)
               
        

        #=======================================================================
        # legend = 
        # # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        # frame = legend.get_frame()
        # frame.set_facecolor('0.90')
        #=======================================================================
        
        #-------------------------------------------------- plt.subplot(2, 1, 1)
        #----------------------------- plt.plot(clusters_no, k_costs , label= m)
        #----------------------------------------------- plt.xticks(clusters_no)
        #-------------------------------------- plt.xlabel('Clusters number K ')
        #---------------------------------------------------- plt.ylabel('Cost')
        #---------------------------- plt.legend(loc='upper right', shadow=True)
#------------------------------------------------------------------------------ 
        #-------------------------------------------------- plt.subplot(2, 1, 2)
        #------------------------------- plt.plot(clusters_no, k_std , label= m)
        #----------------------------------------------- plt.xticks(clusters_no)
        #-------------------------------------- plt.xlabel('Clusters number K ')
        #-------------------------------------- plt.ylabel('Standard Deviation')
             
        
    plt.show()
    
    
    
    
if __name__ == "__main__":
    c1 = load_data_1b("./data1b/C2.txt")
    run_analysis()
    #result = kmeans(c1, int(sys.argv[1]), sys.argv[2])
    #scatterplot(result)
