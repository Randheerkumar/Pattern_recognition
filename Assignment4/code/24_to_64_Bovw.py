'''
pattern recognition
Assignment : 3

This is for generating 64 dimensional bag of visual words vectors from
 24-dimensional vector using k-means clustering


'''

from __future__ import print_function
import numpy as np 
import glob
import imageio
import m_k_means
import pickle

#counting the numbervof data points****************************************************************************
def count(img):
	file=open(img,"r")
	n=0;
	for line in file:
		n+=1
	file.close()
	return(n);

#*************************************************************************************************************
c1="train/movie_theater_indoorF"
c2="train/rock_archF"
c3="train/valleyF"

# varibale initalization *************************************************************************************
k=32        #number of cluster
d=24        # number of feature of a vector
mean=np.zeros((k,d))        # cluster center

#**********************************************************************************************************
N=0
for file in glob.glob(c1+"/*.txt"):           #counting number of vectors in a image
	n=count(file)
	N +=n
for file in glob.glob(c2+"/*.txt"):
	n=count(file)
	N +=n	
for file in glob.glob(c3+"/*.txt"):
	n=count(file)
	N +=n	

#**********************************************************************************************************
print("N=",N)	
x=np.zeros((N,24))	
n=0
for file in glob.glob(c1+"/*.txt"):           #string all vectors in a x array;
	n_file=open(file,"r")
	for line in n_file:
		a=line.split()
		for j in range(len(a)):
			x[n][j]=float(a[j])
		n+=1
	n_file.close()
	
for file in glob.glob(c2+"/*.txt"):          #string all vectors in a x array;
	n_file=open(file,"r")
	for line in n_file:
		a=line.split()
		for j in range(len(a)):
			x[n][j]=float(a[j])
		n+=1
	n_file.close()

for file in glob.glob(c3+"/*.txt"):          #string all vectors in a x array;
	n_file=open(file,"r")
	for line in n_file:
		a=line.split()
		for j in range(len(a)):
			x[n][j]=float(a[j])
		n+=1
	n_file.close()

print(x.shape)

#**********************************************************************************************
m_k_means.parameter(x,k,d,N,mean);
print(mean)

output="mean_32/mean.pkl"

with open(output,'wb') as f:
	pickle.dump(mean,f)


