# Coding --  utf-8
# Face-clustering : K-means clustering for high dimention data.
# python code
# required package: Anaconda
# Author: kewang chen, 06/23/2018
# Contact: kewang.chen@uvm.edu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.cluster import KMeans
import csv
from cycler import cycler

# face_data is preprocessed data. (这里只是删除了有部分数据缺失的样本)
# data size: 154 * 16. (154个样本, 每个有16条特征线的长度值 )  
results = []
tests=[]
with open("face_data.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) 
    # change contents to floats
    for row in reader: # each row is a list
        results.append(row)
# change the list to array
data=np.array(results)
D1=[row[0] for row in data]
D2=[row[1] for row in data]
D5=[row[4] for row in data]
D10=[row[9] for row in data]
D12=[row[11] for row in data]
D13=[row[12] for row in data]
D14=[row[13] for row in data]
D15=[row[14] for row in data]

#plt.plot(D3,'+k')
X_data=np.array([D1,D2,D5,D10,D12,D13,D14,D15]);X=X_data.T

# Initializing KMeans
# face_class means how many classes(要分多少类)
# we know best k by calculate silhouette_value: 
face_class=6
kmeans = KMeans(n_clusters=face_class,max_iter=30000,random_state=10)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
print(labels)
with open('out3.txt', 'w') as f1:
        for x in labels:
                print(x,file=f1)
f1.close()
# Getting the cluster centers
C = kmeans.cluster_centers_
# print the center of each class
print('the center of each class 0~ %i'%(face_class-1))
with open('center2.txt', 'w') as f4:
        for x in C:
                print(x,file=f4)
f4.close()
print(C)


# Classification for new data. (根据最影响分类的几种特征值来预测脸型)
tests2=[]
with open("face_data_test.csv") as csvfile2:
    reader = csv.reader(csvfile2, quoting=csv.QUOTE_NONNUMERIC) 
    # change contents to floats
    for row in reader: # each row is a list
        tests2.append(row)
# change the list to array
data_test=np.array(tests2)
D1_test=[row[0] for row in data_test]
D2_test=[row[1] for row in data_test]
D5_test=[row[4] for row in data_test]
D10_test=[row[9] for row in data_test]
D12_test=[row[11] for row in data_test]
D13_test=[row[12] for row in data_test]
D14_test=[row[13] for row in data_test]
D15_test=[row[14] for row in data_test]

Y_data=np.array([D1_test,D2_test,D5_test,D10_test,D12_test,D13_test,D14_test,D15_test]);Y=Y_data.T
print('New samples need to be classified:')
print(Y)
print('the classification results of new samples are:')
labels3=kmeans.predict(Y)
with open('out5.txt', 'w') as f2:
        for x in labels3:
                print(x,file=f2)
f2.close()
print(labels3)
