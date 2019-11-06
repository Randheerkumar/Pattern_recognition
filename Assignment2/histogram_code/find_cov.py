
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


def find_cov_matrix(x,n,k,d,znk,Pi_k,mean,cov_mat):

	x_znk=np.zeros(k);
	print(x[1][0],x[1][1])
	print(mean[0][0],mean[0][1])
	temp=np.subtract(x[1],mean[0])
	print(temp[0]," ... ",temp[1])
	for i in range(n):
		#print("x=",x[i][1])
		for j in range(n):
			if znk[i][j]==1:
				cov_mat[j]=np.add(cov_mat[j],np.matmul(np.transpose(np.subtract(x[i],mean[j])),np.subtract(x[i],mean[j])))
				x_znk[j]+=1;
				#print("cov mat is for",j, cov_mat[j][0][0], " ",cov_mat[j][0][1])
				break;
	for j in range(k):
		cov_mat[j]=cov_mat[j]/x_znk[j];
		#print("cov mat is", cov_mat[j][0][0], " ",cov_mat[j][0][1])
		Pi_k[j]=x_znk[j]/n;			

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
				#print("ans=",ans)
				cov_mat[j]=np.add(cov_mat[j],ans)
				#test[:]=np.matmul(np.transpose(temp),temp)
				x_znk[j] +=1
			#print("mean & x is:",mean[j],x[i])	
			#print("temp is:",temp)
			#print("cov mat is:(jth)",cov_mat[j])
			#print("test:",test)	
	for j in range(0,k):
		#print(j,x_znk[j])
		for r in range(d):
			for c in range(d):
				if r!=c:
					cov_mat[j][r][c]=0;
				
		if x_znk[j]!=0:					
			cov_mat[j]=cov_mat[j]/x_znk[j]
			cov_mat[j] /=1			
		Pi_k[j]=x_znk[j]/n
	



	



