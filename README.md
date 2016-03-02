# Assignment 1a
## About the project
This code is the python3 implementation of a nearest neighbor classifier for hand-written digit recognition. We also include a wrapper of Scikit Learn's k-neighbor classification for comparison and validation.

## Running
To try our version of nearest neighbor classifier, simply run:

```bash
python3 mnist_dataloader.py
```
To try the Scikit Learn version, run:

```bash
python3 mnist_dataloader.py scikit
```

## Result
The resulting confusion matrix and accuracy are stored in *exercise1-result.txt*.

The ground truths and predicted labels are stored in *label.txt* and *predicted.txt* respectively.

#Assignment 1b
## About the project
This code is the python implementation of a k-means and a k-median clustering algorithm. We also include a wrapper of Scikit Learn's k-means (in the file ```gonzales.py```) for comparison and validation.

## Running
To try k-means, simply run:

```bash
python kmeans.py <number of clusters> method
```
where method can be ```firstk```, ```random```, ```kmeans++```, ```gonz```.
To try the k-median , run:

```bash
python kmedians.py <number of cluster> method distance_metric 
```
where method can be ```firstk```, ```random```, ```kmeans++```, ```gonz```, and distance_metric can be ```eu``` or ```mat```.

#Assignment 1b
## About the project
This code is the python3 implementation of a nearest neighbor classifier for hand-written digit recognition, using Random Projections as the Dimensionality Reduction mean. 

## Running
To try our version of nearest neighbor classifier, simply run:

```bash
python3 mnist_dataloader.py k
```
where k is the number of dimensions we want to reduce to.

## Result
The ground truths and predicted labels are stored in *label_k.txt* and *predicted_k.txt* respectively.
