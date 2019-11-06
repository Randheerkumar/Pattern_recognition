
import matplotlib.pyplot as plt 
import numpy as np


'''
this function is for multiplication of transpose of a matrix by itself
'''
def one_d_matmul(diff,ans,d):
	for i in range(0,d):
		for j in range(0,d):
			ans[i][j]=diff[i]*diff[j]


def f_plot(znk,x,k,size):
	zx=np.zeros((k,20000))
	zy=np.zeros((k,20000))
	for i in range(0,size):
		for  j in range(0,k):
			if(znk[i][j]==1):
				zx[j][i]=x[i][0]
				zy[j][i]=x[i][1]
				break;

	plt.plot(zx[0],zy[0],'ro')
	plt.plot(zx[1],zy[1],'bo')	
	if(k > 2):
		plt.plot(zx[2],zy[2],'yo')
	if(k > 3):
		plt.plot(zx[3],zy[3],'go')
		
	plt.show()	
	

def cov_mat_new(x,n,k,d,znk,Pi_k,mean,cov_mat):
	temp=np.zeros(d)
	test=np.zeros((d,d))
	ans=np.zeros((d,d))
	x_znk=np.zeros(k)
	for j in range(0,k):
		for i in range(0,n):
			if znk[i][j]==1:
				temp[:]=np.subtract(x[i],mean[j])
				#print("ini cov:",cov_mat[j])
				one_d_matmul(temp,ans,d)
				cov_mat[j]=np.add(cov_mat[j],ans)
				#test[:]=np.matmul(np.transpose(temp),temp)
				x_znk[j] +=1
			#print("mean & x is:",mean[j],x[i])	
			#print("temp is:",temp)
			#print("cov mat is:(jth)",cov_mat[j])
			#print("test:",test)	
	for j in range(0,k):
		cov_mat[j]=cov_mat[j]/x_znk[j]
		Pi_k[j]=x_znk[j]/n
	



	



