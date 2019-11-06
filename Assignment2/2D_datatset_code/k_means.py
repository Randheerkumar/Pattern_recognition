import numpy as np
import matplotlib.pyplot as plt         # for plotting
#import plot

def f_plot(znk,x,k,n):
	zx=np.zeros((k,n))
	zy=np.zeros((k,n))
	for i in range(0,n):
		for  j in range(0,k):
			if(znk[i][j]==1):
				if j==0:
					plt.plot(x[i][0],x[i][1],'ro')
				elif j==1:
					plt.plot(x[i][0],x[i][1],'bo')
				elif j==2:
					plt.plot(x[i][0],x[i][1],'go')
				elif j==3:
					plt.plot(x[i][0],x[i][1],'yo')		

				break;

	plt.show()
		


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
				new += znk[i][j]*(np.matmul(temp_array,np.transpose(temp_array)))
                
	
		#print(new,new)
		return np.log(new)


def parameter(x,k,d,n,mean) :
	temp_array=np.zeros(d)
	for i in range(0,k):
		for f in range(0,d) :
			mean[i][f]=x[i][f]                                #randomly choosing mean for all cluster

	temp=np.zeros(k)		                              
	old=0.0
	new=0.0
	c=0
	while c < 2 or old-new > 0.001 :
		c=c+1
		old=new
		znk=np.zeros((n,k))
		new=0.0
		#print("c is",c)
		for i in range(0,n):
			for j in range(0,k):
				temp_array[:]=np.subtract(mean[j],x[i])
				temp[j]=np.matmul(temp_array,np.transpose(temp_array))
			index=np.argmin(temp)
			znk[i][index]=1		
		new=mean_and_cost(mean,x,znk,k,n,d)
		#print("new is",new,"old=",old)

	#f_plot(znk,x,k,n)
	
	
	return(znk)	

   