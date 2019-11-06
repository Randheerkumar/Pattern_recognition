import numpy as np
import matplotlib.pyplot as plt         # for plotting
#import plot

def dot_prod(d,x):
	ans=0.00
	for i in range(d):
		ans+=(x[i]*x[i]);

	return(ans);

'''
def mean_and_cost(mean,x,znk,k,size,d):
		temp_array=np.zeros(d)
		new=0.0
		for j in range(0,k):    #j represents jth cluster 
			nr=np.zeros(d)      #numerator term          
			dr=0.0000000        #denominator term
			for i in range(0,size):     # f is the fth feature of a vector
				for f in range(0,d): 
					nr[f] += znk[i][j]*x[i][f]
				dr += znk[i][j]	
			if(dr > 0):
				mean[j]=nr/dr
					#mean_y[j]=nry/dr
				#else :
				   # mean[j][f]=0.0
				    #mean_y[j]=0.0

		for j in range(0,k):
			for i in range(0,size) :
				temp_array[:]=np.subtract(mean[j],x[i])
				new += dot_prod(d,temp_array)
                
		print(new)
		#print(new,new)
		return np.log(new)
'''
def mean_and_cost(mean,x,znk,k,size,d):
	new=0.0
	for j in range(0,k):    #j represents jth cluste 
		k_sum=np.zeros(d); 
		k_s=0.000;
		for i in range(0,size):     # f is thefth feature of a vector
			if znk[i][j]==1:
				k_sum=np.add(k_sum,x[i])
				k_s+=1
		#print(k_sum)
		#print(k_s)
		if k_s!=0.00:
			mean[j]=k_sum/k_s;
		

	for j in range(0,k):
			for i in range(0,size) :
				temp_array=np.zeros(d)
				if znk[i][j]==1:
					temp_array=np.subtract(mean[j],x[i])
					new += dot_prod(d,temp_array)
	#print(new)
	#print(new,new)
	return np.log(new)



def parameter(x,k,d,n,mean) :
	#temp_array=np.zeros(d)
	step=int(n/k);
	j=0;i=0;
	for j in range(k):
		mean[j]=x[j+step-1];
		i+=1

	#for j in range(0,k):
	#	mean[j]=x[j]                           #randomly choosing mean for all cluster

	#print("mean",mean)
	temp=np.zeros(k)		                              
	old=0.0
	new=0.0
	c=0
	while c < 2 or abs(old-new) > 0.001 :
		c=c+1
		old=new
		znk=np.zeros((n,k))
		new=0.0
		#print("c is",c)
		for i in range(0,n):
			#print(x[i])
			for j in range(0,k):
				temp_array=np.zeros(d)
				temp_array=np.subtract(mean[j],x[i])
				temp[j]=dot_prod(d,temp_array)
			index=np.argmin(temp)
			znk[i][index]=1	
			#print(znk[i])	
		new=mean_and_cost(mean,x,znk,k,n,d)
		print("new is",new,"old=",old)

	
	
	return(znk)	

   