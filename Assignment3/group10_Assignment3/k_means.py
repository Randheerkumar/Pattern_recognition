import numpy as np
from tqdm  import tqdm
import math
import matplotlib.pyplot as plt         # for plotting

#************************************************************************************************************************************
# function returns the dot product of of a vector with itself
def dot_prod(d,x):
	ans=0.00
	for i in range(d):
		ans+=(x[i]*x[i]);

	return(math.sqrt(ans));


#***********************************************************************************************************************************
#this function updates the mean of all k clusters given  Znk and caluclate the log of distortion function
def mean_and_cost(mean,x,znk,k,size,d):
	temp_array=np.zeros(d)                                              #this vector stores the differece b/w mean and a data point 
	new=0.0                                                             #this stores the value distortion function
	for j in range(0,k):    											#j represents Jth cluster 
		nr=np.zeros(d)    												#this stores the sum of all datapoints in jth clusters         
		dr=0.0000000      												#total number of data points in Jth cluster
		for i in range(0,size):     									# i ranges for all the data points 
			nr=np.add(nr,znk[i][j]*x[i])                                #storing the sum of data points in Jth clusters
			dr += znk[i][j]	                                            #storing the total data points in Jth clusters
		if(dr > 0):
			mean[j]=nr/dr                                               # updating the mean of Jth clusters
	print("mean_and_cost")
	for j in range(0,k):
		for i in range(0,size) :
			temp_array[:]=np.subtract(mean[j],x[i])
			new+=znk[i][j]*dot_prod(d,temp_array)
			#new += znk[i][j]*(np.matmul(temp_array,np.transpose(temp_array)))  #calculating the distortition function
            
	return np.log(new)                                                  # taking the log of distortion function and returing it


#**********************************************************************************************************************************
#this method is apply k-means clustering on the given data and calculating centers of the clusters
def parameter(x,k,d,n,mean) :
	temp_array=np.zeros(d)	                                            # stores the differnce b/w mean a cluster and datapoint
	for i in range(0,k):
		mean[i]=x[i]                                                    #randomly choosing mean for all k clusters
		                              	 
	temp=np.zeros(k)                                                    #this stores the distance of datapoint from all the k clusters
	temp_array=np.zeros((k,d))                                          #stores the differnece b/w datapont x and each cluster
	l_gy=np.zeros(n);                                                   #stores the log of distortion funnction in each iteration

	l_g1=0;		                              
	old=0.0
	new=0.0
	c=0
	while c < 2 or abs(old-new) > 0.001 :                               #loop goes until the distortion function converges                            
		c=c+1 
		old=new
		znk=np.zeros((n,k))                                             # this variable stores which data points lies which clusters
		new=0.0
		#print("c is",c)
		for i in range(0,n):                                            # for all datapoints
			for j in range(0,k):
				temp_array[j]=np.subtract(mean[j],x[i])                 # difference of points x to all the mean  of Jth clusters
				temp[j]=dot_prod(d,temp_array[j])                       # finding distance b/w x and mean of Jth clusters
			index=np.argmin(temp)                                       # findin that which clusters x belongs to
			znk[i][index]=1		
		new=mean_and_cost(mean,x,znk,k,n,d)                             # here we call the function for updating the means of all clusters 
	
		l_gy[l_g1]=new;
		l_g1+=1
		print("new is",new,"old=",old)
		#print("in loop for k means")
	
	# below code plots the graph b/w log of distortion function vs no of iteration
	i=0;
	while l_gy[i]!=0.0:
		i+=1;
	l_g1=i;
	x1=np.zeros(l_g1);
	y1=np.zeros(l_g1);
	i=0;
	while l_gy[i]!=0.0:
		x1[i]=i+1;
		y1[i]=l_gy[i];
		i+=1;

	plt.plot(x1,y1,color='cornflowerblue',linestyle='-',marker='o')	
	plt.legend()
	plt.show();

	return(znk)	                                                          # here we return the Znk

   