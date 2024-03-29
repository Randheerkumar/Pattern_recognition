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

#**************************************************************************************************************
def dot_prod(d,x):
	ans=0.00
	for i in range(d):
		ans+=(x[i]*x[i]);

	return(ans);
#loading the data into the variable****************************************************************************
def bov_data(img,mean,k,d,outfile):
	bov=np.zeros(k)
	temp=np.zeros(k)
	x=np.zeros(d)

	f=open(outfile,"a")
	file=open(img,"r")
	for line in file:
		a=line.split()
		for j in range(len(a)):
			x[j]=float(a[j])

		for j in range(0,k):
			temp_array=np.zeros(d)
			temp_array=np.subtract(mean[j],x)
			temp[j]=dot_prod(d,temp_array)
		index=np.argmin(temp)
		bov[index]+=1	

	for j in range(k):
		f.write("%d "%bov[j])
	f.write("\n")		
	file.close()
	f.close()
	#return(x[n-1])



#trainig file********************************************************************************************************
c1="train/movie_theater_indoorF"
c2="train/valleyF"
c3="train/rock_archF"



#test file***************************************************************************************************
tc1="test/movie_theater_indoorTF"
tc2="test/valleyTF"
tc3="test/rock_archTF"

inputt="mean_32/mean.pkl"
#variable declaration starts*************************************************************************************
n1=0;n2=0;n3=0
d=24
k=32

#class1 starts********************************************************************************************
outfile1="32/train/movie.txt"
outfile1t="32/test/movie.txt"

mean=np.zeros((k,d))
with open(inputt,'rb') as f1:
	mean=pickle.load(f1)

for file in glob.glob(c1+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	bov_data(file,mean,k,d,outfile1)

for file in glob.glob(tc1+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	bov_data(file,mean,k,d,outfile1t)
#class2 starts*******************************************************************************************

outfile2="32/train/valley.txt"
outfile2t="32/test/valley.txt"
mean2=np.zeros((k,d))
#with open('Bov/cl2_k_means','rb') as f2:
#	mean2=pickle.load(f2)

for file in glob.glob(c2+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	bov_data(file,mean,k,d,outfile2)

for file in glob.glob(tc2+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	bov_data(file,mean,k,d,outfile2t)	


#class3 starts*********************************************************************************************

outfile3="32/train/rock.txt"
outfile3t="32/test/rock.txt"
mean3=np.zeros((k,d))
#with open('Bov/cl3_k_means','rb') as f3:
#	mean3=pickle.load(f3)

for file in glob.glob(c3+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	bov_data(file,mean,k,d,outfile3)

for file in glob.glob(tc3+"/*.txt"):                                          # reading each images from the source folder                                                         #destination file where i need to output the feature vector 
	bov_data(file,mean,k,d,outfile3t)	

