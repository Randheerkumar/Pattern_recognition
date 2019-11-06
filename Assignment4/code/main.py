'''
pattern recognition
Assignment : 4

'''

from __future__ import print_function
import numpy as np 
import math
from numpy.linalg import eigh
import glob
import pickle
import bays_classifier
import gmm_cluster
import m_k_means
import matplotlib.pyplot as plt
import eigen




#************  varibale initialization   ****************************************************
d=32
l=32
cl=3   #number of classes
k=8#number of cluster

#*******************************************************************************************
c1="32/train/movie.txt"
c2="32/train/rock.txt"
c3="32/train/valley.txt"

tc1="32/test/movie.txt"
tc2="32/test/rock.txt"
tc3="32/test/valley.txt"

#*******************************************************************************************
def find_cluster_zero(znk,clss,n,k):
	for j in range(k):
		summ=0;
		for i in range(n):
			summ+=znk[i][j];
		if(summ>1):
			clss[j]=1;

#******************************************************************************************			
def mainf(x,k,d,n,znk,Pi_k,mean,cov_mat,cn,clss):
	#print("x in mainf",x)
	znk=m_k_means.parameter(x,k,d,n,mean) #clustering using k-means
	#print("in mainf, mean=",mean)
	#print("znk=",znk)
	gmm_cluster.cov_mat_new(x,n,k,d,znk,Pi_k,mean,cov_mat) #intial parameter for gmm
	print(znk);
	print("cov mat i mainf :",cov_mat)
	find_cluster_zero(znk,clss,n,k);
	gmm_cluster.Gmm(x,n,d,k,znk,Pi_k,mean,cov_mat,cn,clss) #gmm clustering


#********************************************************************************************
	


#*******************   dimesion reducing ******************	***********************************



#*************************************************************************
#*****************traning data size ****************************
def count(c):
	i=0
	file1=open(c,"r")
	for line in file1:
			i=i+1;
	file1.close()
	return i

def normalise(x,n,d):
	summ=0.000
	for i in range(n):
		summ=0.000
		for j in range(d):
			summ +=x[i][j]	
		for j in range(d):
			x[i][j] =x[i][j]/summ		

#*********************************************************************************************
n1=count(c1)
n2=count(c2)
n3=count(c3)

#x1=np.zeros((n1,d))
#x2=np.zeros((n2,d))
#x3=np.zeros((n3,d))


x1=np.zeros((n1,d))
a1=np.zeros((n1,l))
mean1=np.zeros((k,l))
znk1=np.zeros((n1,k))
Pi_k1=np.zeros(k);
cov_mat1=np.zeros((k,l,l))


x2=np.zeros((n2,d))
a2=np.zeros((n2,l))
mean2=np.zeros((k,l))
znk2=np.zeros((n2,k))
Pi_k2=np.zeros(k);
cov_mat2=np.zeros((k,l,l))

x3=np.zeros((n3,d))
a3=np.zeros((n3,l))
mean3=np.zeros((k,l))
znk3=np.zeros((n3,k))
Pi_k3=np.zeros(k);
cov_mat3=np.zeros((k,l,l))



#training data size **************************************************************


tn1=count(tc1)
tn2=count(tc2)
tn3=count(tc3)

print("n1,n2,n3",tn1,tn2,tn3)
tx1=np.zeros((tn1,d))
tx2=np.zeros((tn2,d))
tx3=np.zeros((tn3,d))

ta1=np.zeros((tn1,l))
ta2=np.zeros((tn2,l))
ta3=np.zeros((tn3,l))



#************************************************************************************
def store(c,x):
	i=0
	file=open(c,"r")
	for line in file:
		a=line.split()
		for j in range(d):
			x[i][j]=float(a[j])
		i=i+1

#storing the training and testing  data **********************************************
store(c1,x1)
store(c2,x2)
store(c3,x3)

store(tc1,tx1)
store(tc2,tx2)
store(tc3,tx3)

#************************************************************************************
#print(x1,x2,x3)
#print(x1.shape)
n=n1+n2+n3
x=np.zeros((n,d))
j=0
for i in range(n1):
	x[j]=x1[i]
	j+=1
for i in range(n2):
	x[j]=x2[i]
	j+=1
for i in range(n3):
	x[j]=x3[i]
	j+=1


cov_matrix=np.zeros((d,d))
eigen_value=np.zeros(d)
eigen_vec=np.zeros((d,d))

#print("cov mat in main",cov_matrix)
normalise(x,n,d)
normalise(x1,n1,d)
normalise(x2,n2,d)
normalise(x3,n3,d)
normalise(tx1,tn1,d)
normalise(tx2,tn2,d)
normalise(tx3,tn3,d)

cov_matrix=eigen.pca(x,d,n)
eigen_value,eigen_vec=eigh(cov_matrix)
#print("eigen value in main :",eigen_value)
#eigen.print_f(eigen_value,d)
#print(eigen_value);
#eigen.check(x,d,n)


eigen.projection(x1,a1,d,l,n1,eigen_vec)
eigen.projection(x2,a2,d,l,n2,eigen_vec)
eigen.projection(x3,a3,d,l,n3,eigen_vec)
eigen.projection(tx1,ta1,d,l,tn1,eigen_vec)
eigen.projection(tx2,ta2,d,l,tn2,eigen_vec)
eigen.projection(tx3,ta3,d,l,tn3,eigen_vec)

#plt.plot(a1,a2,'ro')
#plt.show()

#print("a1 is",a1)
#print("a2 is",a2)
#print("a3 is",a3)

cls1=np.zeros(k);
cls2=np.zeros(k);
cls3=np.zeros(k);
mainf(a1,k,l,n1,znk1,Pi_k1,mean1,cov_mat1,0,cls1);
#find_cluster_zero(znk1,cls1,n1,k);	
mainf(a2,k,l,n2,znk2,Pi_k2,mean2,cov_mat2,1,cls2)
#find_cluster_zero(znk2,cls2,n2,k);
mainf(a3,k,l,n3,znk3,Pi_k3,mean3,cov_mat3,2,cls3)

print("cov_mat1 :",cov_mat1)
print("cov_mat2 :",cov_mat2)
print("cov_mat3 :",cov_mat3)


confusion_m=np.zeros((cl,cl))
bays_classifier.class_f(ta1,tn1,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,l,0,confusion_m,cls1,cls2,cls3)
#print(confusion_m)
bays_classifier.class_f(ta2,tn2,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,l,1,confusion_m,cls1,cls2,cls3)
#print(confusion_m)
bays_classifier.class_f(ta3,tn3,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,l,2,confusion_m,cls1,cls2,cls3)
print(confusion_m)

'''
file=open("ans.txt","a");
file.write(k);
file.write("\n");
file.write(l);
file.write("\n");

file.write(confusion_m);
file.write("\n");
file.close();'''