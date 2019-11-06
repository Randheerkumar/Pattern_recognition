'''
Pattern recognition cs669

lab 2

Group10

'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt  
import imageio                        # for find_covting
import pickle
import find_cov
import k_means
import gmm_cluster
import test

#here we calculate the number of data points in a file and dimension of the data
def find_dim_and_num_of_data(c1,dim):
	i=0                    
	file1=open(c1,"r")
	for line in file1:
		i=i+1;
	file1.close()
	n=i                     #number of data points in the file 

	i=0;
	file1=open(c1,"r")	
	for line in file1:
		a=line.split()
		for j in range(len(a)):
			i=i+1
		break;	
	file1.close()
	d=i;
	dim[0]=int(d);dim[1]=int(n);
	return(dim)

#here we fetch the data and store it into  variable x
def fetch_data(c1,x):
	file1=open(c1,"r")                              # taking input from file
	i=0;
	for line in file1:
		a=line.split()
		for j in range(len(a)):
			x[i][j]=float(a[j])

		i=i+1
		#print(i)  
	file1.close()	
	                 


#here the main program starts 
k=int(raw_input("enter number of clusters:"))
c1=raw_input("enter file name:")

dim=np.zeros(2)
find_dim_and_num_of_data(c1,dim)
d=int(dim[0])
n=int(dim[1]);
print(n,d)
#here we define various variables
x=np.zeros((n,d))            # for storing the data points 
mean=np.zeros((k,d))         #for storing the cetres of all k clusters
znk=np.zeros((n,k))   
Pi_k=np.zeros(k);     
cov_mat=np.zeros((k,d,d))    # for  covariance matrices of all the clusters

# here we call fetch_data function to store the data in variable x from the file
fetch_data(c1,x)


# here we call the parameters function of k_means file to extimate  the cluster centres and data in each cluster 
znk=k_means.parameter(x,k,d,n,mean)
with open('k_means_mean','wb') as k_means:
	pickle.dump(mean,k_means)


#file="Test/103.png"
#pic= imageio.imread(file)

#test.test_img(pic,file,7,mean)


#find_cov.find_cov_matrix(x,n,k,d,znk,Pi_k,mean,cov_mat)
#test.test_img(pic,file,7,mean)

find_cov.cov_mat_new(x,n,k,d,znk,Pi_k,mean,cov_mat)


gmm_cluster.Gmm(x,n,d,k,znk,Pi_k,mean,cov_mat)

with open('gmm_mean','wb') as g_mean:
	pickle.dump(mean,g_mean)

with open('gmm_cov_mat','wb') as g_cov_mat:
	pickle.dump(cov_mat,g_cov_mat)

with open('gmm_pi_k','wb') as g_pi_k:
	pickle.dump(Pi_k,g_pi_k)

#test.test_img(pic,file,7,mean)


