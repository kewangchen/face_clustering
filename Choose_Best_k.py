
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
from sklearn.metrics import silhouette_samples, silhouette_score
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
range_n_clusters = [6, 7, 8, 9, 10]

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)


    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)