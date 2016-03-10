from openml.apiconnector import APIConnector
import numpy as np
from pandas import Series,DataFrame
import pandas as pd

import random
import matplotlib.pyplot as plt
import re
import sklearn.tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO  
import dot_parser
import pydot
from sklearn import tree
from sklearn.externals.six import StringIO 
from IPython.display import Image  
from sklearn import cross_validation
from sklearn.tree import  ExtraTreeClassifier
apikey = 'b6da739f426042fa9785167b29887d1a'
connector = APIConnector(apikey=apikey)
dataset = connector.download_dataset(10)
train = dataset.get_dataset()
train = pd.DataFrame(train,columns=['lymphatics','block_of_affere','bl_of_lymph_c','bl_of_lymph_s' ,
                            'by_pass' , 'extravasates' , 'regeneration_of' ,'early_uptake_in' ,'lym_nodes_dimin'
                            ,'lym_nodes_enlar' ,'changes_in_lym' ,'defect_in_node' ,'changes_in_node' ,'changes_in_stru'
                            ,'special_forms' ,'dislocation_of','exclusion_of_no','no_of_nodes_in' , 'class'])
train.rename(columns={'class':'target'}, inplace=True)
target_counts = train.target.value_counts()
target_counts
train.target.hist(bins=4)
plt.show()


target = train['target']
X = train.iloc[:,:-1]

#feature importance by Random Forest 
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, target)
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values(inplace=True)
ax = feat_imp.tail(19).plot(kind='barh', figsize=(13,10), title='Feature importance')

plt.show()

#feature importance by decision tree 

clf = tree.DecisionTreeClassifier(min_samples_leaf=3,max_depth=8)
clf = clf.fit(X, target)

feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
feat_imp.sort_values(inplace=True)
ax = feat_imp.tail(19).plot(kind='barh', figsize=(13,10), title='Feature importance by Decision tree')

plt.show()

#removing the small classes
train = train.query('target == 1 or target==2')
target = train['target']
X = train.iloc[:,:-1]


clf = tree.DecisionTreeClassifier(min_samples_leaf=3,max_depth=8)
clf = clf.fit(X, target)

scores = cross_validation.cross_val_score(clf, X, target, cv=5)

print "scores of decision tree with depth max depth 8 and min sample leaf 3"  , scores
print scores.mean()
print scores.std()


dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=X.columns,  
                        class_names=[str('metastases'),str('malign_lymph')],  
                         filled=True, rounded=True, 
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  






clf = tree.DecisionTreeClassifier(min_samples_leaf=5,max_depth=6)
clf = clf.fit(X, target)

scores = cross_validation.cross_val_score(clf, X, target, cv=5)

print "scores of decision tree with depth max depth 6 and min sample leaf 5"  , scores
print scores.mean()
print scores.std()


dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=X.columns,  
                        class_names=[str('metastases'),str('malign_lymph')],  
                         filled=True, rounded=True, 
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  


print "Extra tree classifier"
clf = sklearn.tree.ExtraTreeClassifier(min_samples_leaf=3 ,max_depth=8 )
scores = cross_validation.cross_val_score(clf, X, target, cv=5)
print scores
print scores.mean()

print scores.std()


clf = sklearn.ensemble.ExtraTreesClassifier(n_estimators=10 ,min_samples_leaf=3 ,max_depth=8 )
scores = cross_validation.cross_val_score(clf, X, target, cv=5)
scores

