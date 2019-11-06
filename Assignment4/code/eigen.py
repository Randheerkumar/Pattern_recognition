'''
lab 4

'''

import numpy as np 
from numpy.linalg import eigh
import matplotlib.pyplot as plt


#********** mean calculation ************************

def find_mean(x,n,d):
		mean=np.zeros(d)
		summ=np.zeros(d)
		for j in range(n):
			summ +=x[j]
		mean=summ/n	
		return mean

#****************************************

def dot_product(a,b,d):
	#print("shape of eigen vector:",a.shape)
	#print("shape of y:",b.shape)
	#print(".............")
	summ=0.0
	for i in range(d):
		summ +=a[i]*b[i]
	return summ	

def one_d_matmul(diff,ans,d):
	for i in range(0,d):
		for j in range(0,d):
			ans[i][j]=diff[i]*diff[j]


#********this function gives covariance matrix of origial data  *************
def pca(x,d,N):   
	x_covmat=np.zeros((d,d))                    
	mean=np.zeros(d)
	mean=find_mean(x,N,d)
	x_sub=np.zeros(d)
	for i in range(N):
		x_sub=np.subtract(x[i],mean)
		ans=np.zeros((d,d))
		one_d_matmul(x_sub,ans,d)
		x_covmat =np.add(x_covmat,ans)
	x_covmat /=N
	return x_covmat
	#print("in pca :",x_covmat)

#**********************projection of data **************
def projection(x,a,d,l,n,eigen_vec):
	mean=np.zeros(d)
	mean=find_mean(x,n,d)
	for i in range(n):
		for j in range(l):
			a[i][j]=dot_product(eigen_vec[:,d-j-1],x[i]-mean,d)
    
    
#***********  pritntign eigen vector     ****************************	

def print_f(eigen_value,d):
	x=np.zeros(d)
	for i in range(d):
		#print("eigen value:",eigen_value[i])
		x[i]=i;
		i+=1
	plt.plot(x,eigen_value,'ro')
	plt.show()		

#********************for debuging *********
def check(x,l,n):
	y=np.zeros((n,l))
	mean_x=np.zeros(l)
	#print("in check:shape is:",x[0].shape,mean_x.shape)
	mean_x=find_mean(x,n,l)
	mean_y=np.zeros(l)
	for i in range(n):
		y[i]=x[i]-mean_x
	mean_y=find_mean(y,n,l)
	print("mean of y",mean_y)