'''
Pattern recognition cs669

lab 2

Group10

'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt                          # for plotting
import pickle
import glob
import find_cov
import k_means
import gmm_cluster
import bays_classifier

xg=np.zeros((3,100))
#counting the numbervof data points****************************************************************************
def count(img):
	file=open(img,"r")
	n=0;
	for line in file:
		n+=1
	file.close()
	return(n);

#loading the data into the variable****************************************************************************
def load_data(img,x,d):
	mean=np.zeros(d)
	n=0;
	file=open(img,"r")
	for line in file:
		a=line.split()
		for j in range(len(a)):
			x[n][j]=float(a[j])
		n+=1

	file.close()
	#return(x[n-1])

#estimating the pararmeter**************************************************************************************
def mainf(x,k,d,n,znk,Pi_k,mean,cov_mat,cn):
	znk=k_means.parameter(x,k,d,n,mean) #clustering using k-means

	find_cov.cov_mat_new(x,n,k,d,znk,Pi_k,mean,cov_mat) #intial parameter for gmm

	#print(Pi_k)
    
	gmm_cluster.Gmm(x,n,d,k,znk,Pi_k,mean,cov_mat,cn,xg) #gmm clustering	


#trainig file********************************************************************************************************
c1="train/movie_theater_indoorF"
c2="train/valleyF"
c3="train/rock_archF"



#test file***************************************************************************************************
tc1="test/movie_theater_indoorTF"
tc2="test/valleyTF"
tc3="test/rock_archTF"


#variable declaration starts*************************************************************************************
n1=0;n2=0;n3=0
d=24
cl=3
k=2

#class1 starts***************************************************************************************
for file in glob.glob(c1+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	n=count(file)
	n1+=n;

x1=np.zeros((n1,d))
mean1=np.zeros((k,d))
#print(mean1)
znk1=np.zeros((n1,k))
Pi_k1=np.zeros(k);
cov_mat1=np.zeros((k,d,d))
i=0;
for file in glob.glob(c1+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	load_data(file,x1,d)


#print(mean1)
#print(mean1[1])
#print(mean1[2])

#class2 starts*******************************************************************************************8
for file in glob.glob(c2+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	n=count(file)
	n2+=n;

x2=np.zeros((n2,d))
mean2=np.zeros((k,d))
znk2=np.zeros((n2,k))
Pi_k2=np.zeros(k);
cov_mat2=np.zeros((k,d,d))


for file in glob.glob(c2+"/*.txt"):                           # reading each images from the source folder                                                             #destination file where i need to output the feature vector 
	load_data(file,x2,d)

#class3 starts*********************************************************************************************

for file in glob.glob(c3+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	n=count(file)
	n3+=n;

x3=np.zeros((n3,d))
mean3=np.zeros((k,d))
znk3=np.zeros((n3,k))
Pi_k3=np.zeros(k);
cov_mat3=np.zeros((k,d,d))

for file in glob.glob(c3+"/*.txt"):                           # reading each images from the source folder                                                             #destination file where i need to output the feature vector 
	load_data(file,x3,d)

#calling main **********************************************************************************************
mainf(x1,k,d,n1,znk1,Pi_k1,mean1,cov_mat1,0)
mainf(x2,k,d,n2,znk2,Pi_k2,mean2,cov_mat2,1)
mainf(x3,k,d,n3,znk3,Pi_k3,mean3,cov_mat3,2)

confusion_m=np.zeros((cl,cl))
'''
#testing data extraction ***********************************************************************************
tn1=np.zeros(50)
i=0;
for file in glob.glob(tc1+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	tn1[i]=count(file)
	i+=1

i=0;
for file in glob.glob(tc1+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	tx1=np.zeros((tn1[i],d))
	load_data(file,tx1,d)
	bays_classifier.class_f(tx1,tn1,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d,0,confusion_m)
	i+=1



'''	
# testing data of calss2**********************************************************************************
'''
tn2=0
for file in glob.glob(tc2+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	n=count(file)
	tn2+=n;
tx2=np.zeros((tn2,d))

for file in glob.glob(tc2+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	load_data(file,tx2,d)

#testing of calss3*******************************************************************************************
tn3=0
for file in glob.glob(tc3+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	n=count(file)
	tn3+=n;
tx3=np.zeros((tn3,d))


for file in glob.glob(tc3+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	load_data(file,tx3,d)	
#starts calling the function ********************************************************************************


'''

confusion_m=np.zeros((cl,cl))

tn1=0
for file in glob.glob(tc1+"/*.txt"): 
	tn1=count(file)
	tx1=np.zeros((tn1,d))
	load_data(file,tx1,d)
	bays_classifier.class_f(tx1,tn1,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d,0,confusion_m)
#print(confusion_m)
tn2=0
for file in glob.glob(tc2+"/*.txt"): 
	tn2=count(file)
	tx2=np.zeros((tn2,d))
	load_data(file,tx2,d)
	bays_classifier.class_f(tx2,tn2,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d,1,confusion_m)
#print(confusion_m)
tn3=0
for file in glob.glob(tc3+"/*.txt"): 
	tn3=count(file)
	tx3=np.zeros((tn3,d))
	load_data(file,tx3,d)
	bays_classifier.class_f(tx3,tn3,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d,2,confusion_m)
print(k)
print(confusion_m)

'''
#print(cov_mat1)
#print(cov_mat2)
#print(cov_mat3)
'''
#bays_classifier.decplot(minx,maxx,miny,maxy,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d)


