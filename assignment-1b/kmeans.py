import numpy as np
from plotting import *

MAX_NUMBER = 99999999999

def load_data_1b(fpath):
    data = []
    f = open(fpath, 'r')
    for line in f:
        words = line.split()
        data.append(words)
    f.close()
    arr = np.array(data, dtype=np.float64)
    return arr[:, 1:]

def cluster_distance(cluster1, cluster2):
    center1 = np.average(cluster1)
    center2 = np.average(cluster2)
    
    return np.linalg.norm(center1 - center2)

def chooseKCenter(data, k):
    seeds = [data[np.random.random_integers(len(data) - 1)]]
    distances = []
    for r in range(1,k):
        for point in data:
            min_dist = MAX_NUMBER
            for i in range(r):
                dist = np.linalg.norm(point - seeds[i])
                if dist < min_dist:
                    min_dist = dist
            distances.append(np.power(min_dist, 2))

        for i in range(len(data)):
            prob = np.divide(distances[i], np.sum(distances))
            if np.random.random_sample() >= prob:
                seeds.append(data[i])
                break

    return seeds

def kmeans(data, k, method):
    if method == 'firstk':
        seeding_points = data[0:k]
    elif method == 'random':
        random_indices = np.random.random_integers(len(data) - 1, size = (1,k)).flatten()
        seeding_points = [data[i] for i in random_indices]
    elif method == 'kmeans++':
        seeding_points = chooseKCenter(data, k)
        

    old_clusters = [[]]
    new_clusters = [[] for i in range(k)]
    old_cost = np.array([0 for i in range(k)])
    new_cost = np.array([1 for i in range(k)])
    #while(not np.array_equal(old_cost, new_cost)):

    while True:
    #for i in range(100):
        print('process next round')
        print('seeding_points', seeding_points)
        old_clusters = new_clusters
        new_clusters = [[] for i in range(k)]
        old_cost = new_cost
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
         
        print('==============')
        print('average 0', np.average(new_clusters[0]))
        print('average 1', np.average(new_clusters[1]))
        print('average 2', np.average(new_clusters[1]))
        new_seeds = [np.average(new_clusters[i]) for i in range(k)]
        seeding_points = new_seeds

        distance = np.sum([cluster_distance(old_clusters[i], new_clusters[i]) for i in range(k)])
        print(distance)
        #for i in range(k):
        #    distance = 0
        #    distance = distance + cluster_distance(old_clusters[i], new_clusters[i])
        
        if distance == 0:
            break

    return new_clusters


if __name__ == "__main__":
    c1 = load_data_1b("./data1b/C2.txt")
    #result = kmeans(c1, 2, 'firstk')
    #result = kmeans(c1, 3, 'random')
    result = kmeans(c1, 3, 'kmeans++')
    scatterplot(result)
