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
	#print(mean)

	find_cov.cov_mat_new(x,n,k,d,znk,Pi_k,mean,cov_mat) #intial parameter for gmm
	print(znk)

	#print(cov_mat[0])


	gmm_cluster.Gmm(x,n,d,k,znk,Pi_k,mean,cov_mat,cn,xg) #gmm clustering	


#trainig file********************************************************************************************************
c1="train/movie_theater_indoor.txt"
c2="train/valley.txt"
c3="train/rock_arch.txt"



#test file***************************************************************************************************
tc1="test/movie_theater_indoor.txt"
tc2="test/valley.txt"
tc3="test/rock_arch.txt"


#variable declaration starts*************************************************************************************
n1=0;n2=0;n3=0
d=32
cl=3
k=4

#class1 starts***************************************************************************************                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
n1=count(c1)

x1=np.zeros((n1,d))
mean1=np.zeros((k,d))
#print(mean1)
znk1=np.zeros((n1,k))
Pi_k1=np.zeros(k);
cov_mat1=np.zeros((k,d,d))
                                      
load_data(c1,x1,d)


#print(mean1)
#print(mean1[1])
#print(mean1[2])

#class2 starts*******************************************************************************************

n2=count(c2)

x2=np.zeros((n2,d))
mean2=np.zeros((k,d))
#print(mean1)
znk2=np.zeros((n2,k))
Pi_k2=np.zeros(k);
cov_mat2=np.zeros((k,d,d))
                                      
load_data(c2,x2,d)


#class3 starts*********************************************************************************************


n3=count(c3)

x3=np.zeros((n3,d))
mean3=np.zeros((k,d))
#print(mean1)
znk3=np.zeros((n3,k))
Pi_k3=np.zeros(k);
cov_mat3=np.zeros((k,d,d))
                                      
load_data(c3,x3,d)


#starts calling the function ********************************************************************************
mainf(x1,k,d,n1,znk1,Pi_k1,mean1,cov_mat1,0)
#with open('Bov/cl1_k_means','wb') as f1:
#	pickle.dump(mean1,f1)


mainf(x2,k,d,n2,znk2,Pi_k2,mean2,cov_mat2,1)
#with open('Bov/cl2_k_means','wb') as f2:
#	pickle.dump(mean2,f2)


mainf(x3,k,d,n3,znk3,Pi_k3,mean3,cov_mat3,2)
#with open('Bov/cl3_k_means','wb') as f3:
#	pickle.dump(mean3,f3)
#print(cov_mat1)
#print(mean1)
#print(cov_mat2)
#print(mean2)
#print(cov_mat3)
#print(mean3)
#plt.legend()
#plt.show()

tn1=count(c1)
tx1=np.zeros((tn1,d))                                      
load_data(tc1,tx1,d)

tn2=count(c2)
tx2=np.zeros((tn2,d))                                      
load_data(tc2,tx2,d)

tn3=count(c3)
tx3=np.zeros((tn3,d))                                      
load_data(tc3,tx3,d)




confusion_m=np.zeros((cl,cl))
bays_classifier.class_f(tx1,tn1,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d,0,confusion_m)
#print(confusion_m)
bays_classifier.class_f(tx2,tn2,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d,1,confusion_m)
#print(confusion_m)
bays_classifier.class_f(tx3,tn3,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d,2,confusion_m)
print(k)
print(confusion_m)

#print(cov_mat1)
#print(cov_mat2)
#print(cov_mat3)
'''
#bays_classifier.decplot(minx,maxx,miny,maxy,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d)

'''

col=(['red','blue','green'])
name=(['movie-theatre','valley','roch'])

for j in range(0,3):
	count=0;
	for i in range(100):
			if xg[j][i]==0:
				break;
			count+=1;
	temp=np.zeros(count)
	epoch=np.zeros(count)
	for i in range(0,count):
		temp[i]=xg[j][i]	
		epoch[i]=i+1
	plt.plot(epoch,temp,color=col[j],label=name[j],linestyle='-',marker='o')
plt.legend()
plt.show()



