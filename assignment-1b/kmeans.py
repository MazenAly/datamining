import numpy as np
from plotting import *
from scipy import spatial
import math
import sys

MAX_NUMBER = math.inf 
import math
import dataloader_1b as mn 

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

    return np.linalg.norm(center1 - center2)

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
    while True:
        print('process next round')
        print('seeding_points', seeding_points)
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
         
        print('==============')
        seeding_points = [np.average(new_clusters[i],0) for i in range(k)]
        distance = np.sum([cluster_distance(old_clusters[i], new_clusters[i]) for i in range(k)])
        print('cost', cost)
        
        if distance == 0:
            break

    return new_clusters


if __name__ == "__main__":
    c1 = load_data_1b("./data1b/C2.txt")
    result = kmeans(c1, int(sys.argv[1]), sys.argv[2])
    scatterplot(result)
