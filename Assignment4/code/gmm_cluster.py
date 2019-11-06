
import numpy as np


def one_d_matmul(diff,ans,d):
	for i in range(0,d):
		for j in range(0,d):
			ans[i][j]=diff[i]*diff[j]


def make_diagonal(cov_matrix,d):
	for i in range(d):
		for j in range(d):
			if i != j:
				cov_matrix[i][j]=0
			else:
				if cov_matrix[i][j]==0.0:
					cov_matrix[i][j]=0.0001	



def G(pos,mu,cov_matrix,d):        
	n = d
	make_diagonal(cov_matrix,d)
	if np.linalg.det(cov_matrix)==0.0:
		print("det is zero",cov_matrix)
	Sigma_det = np.linalg.det(cov_matrix)	
	print("det is",Sigma_det)	
	#if Sigma_det > 1e32 :
		#print(cov_matrix) 
	inv_cov_matrix=np.linalg.inv(cov_matrix)
	
	N = np.sqrt((2*np.pi)**n * Sigma_det)

	# This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
	# way across all the input variables.
	fac = np.einsum('...k,kl,...l->...', pos-mu, inv_cov_matrix, pos-mu)
	return np.exp(-fac / 2)/N 



def Gmm(x,n,d,k,Z_nk,Pi_k,mean,cov_mat,cn,clss):
	print("in,gmm fn ,class no=",cn)
	#print("cov mat is in gmm :",cov_mat)
	ans=np.zeros((d,d))
	diff=np.zeros(d)
	l_new=0.0
	l_old=0.00
	c=0
	epoch=0
	#print("cov mat is :",cov_mat)
	while c < 2 or l_new-l_old>0.001:
		c=c+1
		l_old=l_new
		l_new=0.0000
		total_sum=0.0
		for i in range(n):
			summ=0.0000;
			for j in range(k):
				if clss[j]==1:
					summ=summ+(Pi_k[j]*G(x[i],mean[j],cov_mat[j],d))
			for j in range(k):
				#print("sum is:",summ)
				if clss[j]==1:
					Z_nk[i][j]=(Pi_k[j]*G(x[i],mean[j],cov_mat[j],d))*1.000/summ
			if summ!=0.0:		
				l_new += np.log(summ)
		#print(l_new)	
		if epoch < 100:
			#xg[cn][epoch]=l_new
			#print(l_new)
			epoch+=1

		Pi_Z_nk_sum=np.zeros(k);
		mean_Z_nk_sum=np.zeros((k,d));
		cov_mat_Z_nk_sum=np.zeros((k,d,d))
		for i in range(n):
			for j in range(k):
				if clss[j]==1:
					Pi_Z_nk_sum[j]+=Z_nk[i][j];
					mean_Z_nk_sum[j]=np.add(mean_Z_nk_sum[j],Z_nk[i][j]*x[i]);
					#cov_mat_Z_nk_sum[j]=np.add(cov_mat_Z_nk_sum[j],Z_nk[i][j]*(np.matmul(np.transpose(np.subtract(x[i],mean[j])),np.subtract(x[i],mean[j]))))
		
		for j in range(k):
			if clss[j]==1:
				Pi_k[j]=(Pi_Z_nk_sum[j]/n)
				mean[j]=mean_Z_nk_sum[j]/Pi_Z_nk_sum[j]
			#cov_mat_Z_nk_sum[j]=cov_mat_Z_nk_sum[j]/Pi_Z_nk_sum[j];
		for i in range(n):
			for j in range(k):
				if clss[j]==1:
					diff[:]=x[i]-mean[j]
					one_d_matmul(diff,ans,d)
					cov_mat_Z_nk_sum[j]=np.add(cov_mat_Z_nk_sum[j],Z_nk[i][j]*ans)


		for j in range(k):
			if clss[j]==1:
				cov_mat[j]=cov_mat_Z_nk_sum[j]/Pi_Z_nk_sum[j];
		#print("mean",mean)
	    #print("cov:",cov_mat)
	    
	#print("mean..",mean)
	#print("cov:",cov_mat)

	

def cov_mat_new(x,n,k,d,znk,Pi_k,mean,cov_mat):
	temp=np.zeros(d)
	test=np.zeros((d,d))
	ans=np.zeros((d,d))
	x_znk=np.zeros(k)
	#print("mean before :",mean)
	for j in range(0,k):
		for i in range(0,n):
			if znk[i][j]==1:
				temp[:]=np.subtract(x[i],mean[j])
				#print("temp is",temp)
				#print("ini cov:",cov_mat[j])
				one_d_matmul(temp,ans,d)
				#print("ans is : ",ans)
				#print("ans=",ans)
				cov_mat[j]=np.add(cov_mat[j],ans)
				#test[:]=np.matmul(np.transpose(temp),temp)
				x_znk[j] +=1
			#print("mean & x is:",mean[j],x[i])	
			#print("temp is:",temp)
			#print("cov mat is:(jth)",cov_mat[j])
			#print("test:",test)	
		cov_mat[j]=cov_mat[j]/x_znk[j]		
		Pi_k[j]=x_znk[j]/n
	