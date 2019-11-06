#importing the required Libararies
import numpy as np
import glob
import imageio
import pickle
import matplotlib.pyplot as plt
import math

#*******************************************************************************************************************************************
def dot_product(diff,d):
	dist=0;
	for i in range(d):
		dist+=diff[i]*diff[i];
	return(math.sqrt(dist));	

#******************************************************************************************************************************************
def find_cluster(mean,x,k,d):
	temp=np.zeros(k)
	diff=np.zeros(d)
	for j in range(k):
		diff=np.subtract(mean[j],x);
		temp[j]=dot_product(diff,d);

	index=np.argmin(temp);	
	return(index);


#******************************************************************************************************************************************

def store_quant_data(source_path,source_pathout,mean,k,d):
	x=np.zeros(d)
	file=open(source_path,"r");
	fileout=open(destination,"a+")
	for line in file:
		a=line.split();
		for j in range(len(a)):
			x[j]=float(a[j]);
		index=find_cluster(mean,x,k,d)
		index+=1
		fileout.write("%d "%index)	
	fileout.write("\n");
	fileout.close()				

#*****************************************************************************************************************************************
source_path1="Group10/Train/kA";
source_path2="Group10/Train/kha";
source_path3="Group10/Train/khA";
source_pathout="Group10/Train_m/m_32/";



source_path1t="Group10/Test/kA";
source_path2t="Group10/Test/kha";
source_path3t="Group10/Test/khA";
source_pathoutT="Group10/Test_m/m_32/";





d=39
k=32
mean1=np.zeros((k,d))
with open('m_obsr_sym/m_32.pkl','rb') as f1:
	mean1=pickle.load(f1)


for file in glob.glob(source_path1+"/*.mfcc"):
	destination=source_pathout+"kA"+".txt"
	store_quant_data(file,destination,mean1,k,d)
print("complted",source_path1)

for file in glob.glob(source_path2+"/*.mfcc"):
	destination=source_pathout+"kha"+".txt"
	store_quant_data(file,destination,mean1,k,d)
print("complted",source_path2)


for file in glob.glob(source_path3+"/*.mfcc"):
	destination=source_pathout+"khA"+".txt"
	store_quant_data(file,destination,mean1,k,d)
print("complted",source_path3)


#testing ****************************************************************************************************
for file in glob.glob(source_path1t+"/*.mfcc"):
	destination=source_pathoutT+"kA"+".txt"
	store_quant_data(file,destination,mean1,k,d)
print("complted",source_path1t)

for file in glob.glob(source_path2t+"/*.mfcc"):
	destination=source_pathoutT+"kha"+".txt"
	store_quant_data(file,destination,mean1,k,d)
print("complted",source_path2t)


for file in glob.glob(source_path3t+"/*.mfcc"):
	destination=source_pathoutT+"khA"+".txt"
	store_quant_data(file,destination,mean1,k,d)
print("complted",source_path3t)
		
		










