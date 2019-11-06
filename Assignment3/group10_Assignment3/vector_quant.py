#importing the required Libararies
import numpy as np
import glob
import imageio
import pickle
import matplotlib.pyplot as plt
import k_means

#*******************************************************************************************************************************************
#this function counts  the numbers of data points all obervation of a class  
def find_num_data(source_path):
	n=0;
	for file in glob.glob(source_path+"/*.mfcc"):
		f=open(file,"r")
		for line in f:
			n+=1;
		f.close();	
	return(n);


#*******************************************************************************************************************************************
# this function stores the data into variable
def load_data(source_path,x,n):
	for file in glob.glob(source_path+"/*.mfcc"):
		f=open(file,"r")
		for line in f:
			a=line.split();
			for j in range(len(a)):
				x[n][j]=float(a[j])
			n+=1
		f.close();		
	return(n);


#*********************************************************************************************************************************************
# below are the files where training and testing data is present
source_path1="Group10/Train/kA";
source_path2="Group10/Train/kha";
source_path3="Group10/Train/khA";

source_pathT1="Group10/Test/kA";
source_pathT2="Group10/Test/kha";
source_pathT3="Group10/Test/khA";

#finding the total number of data points for all the trainig classes
n=find_num_data(source_path1)+find_num_data(source_path2)+find_num_data(source_path3)  # total numbers of data points
d=39
k=4                                                                                    # dimension of data points

x=np.zeros((n,d))                                                                      # variables for storing  the data of all observation of all class                                                                 
i=0;  
#here we are storing the data into the variable x                                                                                 
i=load_data(source_path1,x,i)
i=load_data(source_path2,x,i)
i=load_data(source_path3,x,i)


#here considering 8 symbols i.e 8 clustering of the data
mean1=np.zeros((8,d)) 
k_means.parameter(x,8,d,n,mean1)

with open('m_obsr_sym/m_8.pkl', 'wb') as f1:
    pickle.dump(mean1, f1) 

print("completed8")



#here considering 16 symbols i.e 16 clustering of the data
mean2=np.zeros((16,d))  
k_means.parameter(x,16,d,n,mean2)	

with open('m_obsr_sym/m_16.pkl', 'wb') as f2:
    pickle.dump(mean2, f2)

print("completed16")



#here considering 32 symbols i.e 32 clustering of the data
mean3=np.zeros((32,d))  
k_means.parameter(x,32,d,n,mean3)	

with open('m_obsr_sym/m_32.pkl', 'wb') as f3:
    pickle.dump(mean3, f3) 

print("completed32")         