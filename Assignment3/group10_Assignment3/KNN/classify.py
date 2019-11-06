#importing required ibraries
import numpy as np
import glob
import math
from scipy.spatial import distance as ec_dist   


#******************************************************************************************************************************************
#this function returns the length of observation
def find_len_obs(source_file):
	n=0;
	file=open(source_file,"r");
	for line in file:
		n+=1;
	file.close();
	return(n);

#*****************************************************************************************************************************************		
# this function stores the observation into the valriable tx
def load_obs(source_file,tx):
	file=open(source_file,"r");
	i=0;
	for line in file:
		a=line.split();
		for j in range(len(a)):
			tx[i][j]=float(a[j]);	
		i+=1;

#****************************************************************************************************************************************
#this function calculate the disttance between two d demsional vector
def find_dist(train_x,test_x,d):
	dist=0;
	diff=np.zeros(d);
	diff=np.subtract(train_x,test_x);
	for i in range(d):
		dist+=diff[i]*diff[i];
	return(dist);

#**************************************************************************************************************************************		
# this function calculates the dtw distance between two observation sequence
def DTW(train_x,test_x,train_n,test_n,d):
	dtw=np.zeros((train_n,test_n));
	for i in range(train_n):
		for j in range(test_n):
			dtw[i][j]=ec_dist.euclidean(train_x[i],test_x[j]);		

	i=1;
	while i <train_n:
		dtw[i][0]+=dtw[i-1][0];
		i+=1;

	j=1;
	#print("2")
	while j<test_n:
		dtw[0][j]+=dtw[0][j-1];
		j+=1;

	i=1;j=1;
	#print("3")
	while i< train_n:
		j=1;
		while j< test_n:
			dtw[i][j]+=min(dtw[i-1][j],dtw[i][j-1],dtw[i-1][j-1]);
			j+=1;
		i+=1;	

	dist=(dtw[train_n-1][test_n-1])/(train_n*test_n);
	#print("4")
	return(dist);    	
#*****************************************************************************************************************************************
#finding the class from which the given observation belog to
def find_class(k_nn,k):
	k_nn.sort(key=lambda x : x[0] , reverse = False);
	temp=np.zeros(3);
	i=1;
	for x,y in k_nn:
		temp[int(y)]+=1;
		i+=1;
		if(i>k):
			break;
	print("i=",i);
	#print(temp);
	index=np.argmax(temp);
	#print(k_nn);	
	return(index);	

#******************************************************************************************************************************************
# this function takes the observation of 1 class and classify the observation 
def knn_classify(source_patht,source_path1,source_path2,source_path3,confusion_matrix,k,d,cl):

	#confusion_matrix=np.zeros((3,3));
	j=0;
	for file in glob.glob(source_patht+"/*.mfcc"):

		n=find_len_obs(file);
		tx=np.zeros((n,d));
		load_obs(file,tx);
		k_nn=[]
		i=0;
		for train_file1 in glob.glob(source_path1+"/*.mfcc"):
			n1=find_len_obs(train_file1);
			x1=np.zeros((n1,d));
			load_obs(train_file1,x1);
			dist1=DTW(x1,tx,n1,n,d);
			#print(dist1,0);
			k_nn.append([dist1,0]);
			i+=1;
			#if(i>70):
			#	break;

		#print(i);
		for train_file2 in glob.glob(source_path2+"/*.mfcc"):
			n2=find_len_obs(train_file2);
			x2=np.zeros((n2,d));
			load_obs(train_file2,x2);
			dist2=DTW(x2,tx,n2,n,d);
			#k_nn[i][0]=dist;k_nn[i][1]=1;
			#print(dist2,1);
			k_nn.append([dist2,1]);
			i+=1;
		#print(i);
		for train_file3 in glob.glob(source_path3+"/*.mfcc"):
			n3=find_len_obs(train_file3);
			x3=np.zeros((n3,d));
			load_obs(train_file3,x3);
			dist3=DTW(x3,tx,n3,n,d);
			#k_nn[i][0]=dist;k_nn[i][1]=2;
			#print(dist3,2);
			k_nn.append([dist3,2]);
			i+=1; 
		#print(i)
		index=find_class(k_nn,k);
		print(index);
		confusion_matrix[cl][index]+=1;	
		j+=1
		#print(k_nn);
		#print(i);	 
		#break;
		#print(j);
		del k_nn
		#if j>21:
		#	break;
	  