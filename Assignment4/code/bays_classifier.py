
import numpy as np
import gmm_cluster

def make_diagonal(cov_matrix,d):
	for i in range(d):
		for j in range(d):
			if i != j:
				cov_matrix[i][j]=0;
			else:
				if cov_matrix[i][j]==0.00:
					cov_matrix[i][j]=.0001;	


def prob(pos,mu,cov_matrix,d):        
	n = d
	make_diagonal(cov_matrix,d)
	inv_cov_matrix=np.linalg.inv(cov_matrix)
	Sigma_det = np.linalg.det(cov_matrix)
	print(Sigma_det);
	N = np.sqrt((2*np.pi)**n * abs(Sigma_det))

	# This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
	# way across all the input variables.
	fac = np.einsum('...k,kl,...l->...', pos-mu, inv_cov_matrix, pos-mu)

	return np.exp(-fac / 2) / N


def class_f(x,n,mean1,cov1,Pi_k1,mean2,cov2,Pi_k2,mean3,cov3,Pi_k3,k,d,cn,conf,clss1,clss2,clss3):
		
		for i in range(n):
			temp=np.zeros(3)
			for j in range(k):
				if clss1[j]==1 :
					temp[0] += (Pi_k1[j]*prob(x[i],mean1[j],cov1[j],d))
			for j in range(k):
				if clss2[j]==1:
					temp[1] += (Pi_k2[j]*prob(x[i],mean2[j],cov2[j],d))
			for j in range(k):
				if clss3[j]==1:
					temp[2] += (Pi_k3[j]*prob(x[i],mean3[j],cov3[j],d))


			index=np.argmax(temp)
			conf[cn][index]+=1;
			

