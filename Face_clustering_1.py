# Coding --  utf-8
# Face-clustering : K-means clustering for high dimention data.
# python code
# required package: Anaconda
# Author: kewang chen, 06/24/2018
# Contact: kewang.chen@uvm.edu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.cluster import KMeans
import csv

# face_data is preprocessed data. (这里只是删除了有部分数据缺失的样本)
# data size: 800 * 16. (800个样本, 每个有16条特征线的长度值 )  
results = []
tests=[]
with open("face_data.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)
# change the list to array
data=np.array(results)
data_std_1=np.std(data,axis=0)
plt.figure(3)
plt.plot(data_std_1,'--or')
# the following lines are used to indicate the Variances of each feature (最影响分类的几种特征值)
line0=[3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71, 3.71]
line1=[6.63, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62, 6.62]
#line2=[4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9, 4.9]
plt.plot(line0,'--');#plt.plot(line1,'--');
plt.plot(line1,'--k')
plt.title('Variance of each characters')
#labels_kw=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18]
#plt.xticks(line0,labels_kw)
plt.xlabel('Characters');plt.ylabel('Variance')
# Di is the i_th characters 
D1=[row[0] for row in data]
D2=[row[1] for row in data]
D3=[row[2] for row in data]
D4=[row[3] for row in data]
D5=[row[4] for row in data]
D6=[row[5] for row in data]
D7=[row[6] for row in data]
D8=[row[7] for row in data]
D10=[row[9] for row in data]
D12=[row[11] for row in data]
D13=[row[12] for row in data]
D14=[row[13] for row in data]
D15=[row[14] for row in data]
D16=[row[15] for row in data]

plt.figure(2)
# we could try different plot here
plt.plot(D1,'or')
plt.plot(D5,'*b')
plt.title('line1 vs line5')
#plt.plot(D3,'+k')

plt.figure(1)
# we could try different plot here
plt.plot(D14,'or')
plt.plot(D6,'*b')
plt.title('line14 vs line6')

plt.figure(7)
# we could try different plot here
plt.plot(D14,'or')
plt.plot(D7,'*b')
plt.title('line14 vs line7')

plt.figure(6)
# we could try different plot here
plt.plot(D15,'or')
plt.plot(D3,'*b')
plt.title('line15 vs line3')
#plt.plot(D3,'+k')
X_data=np.array([D1,D13,D15]);X=X_data.T
fig = plt.figure(4)
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])

# Initializing KMeans
# face_class means how many classes(要分多少类)
# for example: face_class=10 or 15 or 20
# we could also initialize the cluster centers (聚类中心), here we just use random choice.
# Note: if we set the initial cluster centers as random, 每次运行结果不一定相同 但分类大体一致!
face_class=10
kmeans = KMeans(n_clusters=face_class,max_iter=20000,random_state=1000)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)


with open("kw_result.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(labels)
print(labels)
# Getting the cluster centers
C = kmeans.cluster_centers_
fig = plt.figure(5)
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=labels)
#ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', s=200,c='r')
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
if hasattr(kmeans, 'cluster_centers_'):
        centers = kmeans.cluster_centers_
        center_colors = colors[:len(centers)]
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', s=200, c=center_colors)
plt.title('Clusters=%i'%face_class) 
plt.xlabel('line1')
plt.ylabel('line13') 
ax.set_zlabel('line15')  
plt.show()
# print the center of each class
print("the center of each class are:")
print(C)
# Save the lables in outtxt (分类结果)
result_kw=labels.T
with open('out.txt', 'w') as f:
        for x in labels:
                print(x,file=f)

# Classification for new data. (根据最影响分类的几种特征值来预测脸型)
tests=[]
with open("face_data_test.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        tests.append(row)
# change the list to array
data_test=np.array(tests)
D1_test=[row[0] for row in data_test]
D13_test=[row[12] for row in data_test]
D15_test=[row[14] for row in data_test]
Y_data=np.array([D1_test,D13_test,D15_test]);Y=Y_data.T
print('the classification results of new samples are:')
labels2=kmeans.predict(Y)
print(labels2)
