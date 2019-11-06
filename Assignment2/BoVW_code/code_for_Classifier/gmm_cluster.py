import matplotlib.pyplot as plt
import numpy as np
from decimal import *
#import bigfloat
import find_cov

a=1000000
def inv(cov_mat,d):
	inv_cov=np.zeros((d,d))
	for i in range(d):
		if abs(cov_mat[i][i])<0.00001:
			inv_cov[i][i]=10000
			if cov_mat[i][i] < 0 :
				inv_cov[i][i]=-100000	
			
		elif abs(cov_mat[i][i]) > 100000:
				inv_cov[i][i]=0.00001
				if cov_mat[i][i] < 0 :
					inv_cov[i][i]=-0.00001

		else:
			inv_cov[i][i]=1/cov_mat[i][i]	

	return(inv_cov)

def det(cov_mat,d):
	dett=1
	for i in range(d):
		if cov_mat[i][i]!=0.000:
				dett=dett*np.sqrt(abs(cov_mat[i][i]))
	

	#return max(1,np.sqrt(abs(dett)))
	return dett	


def G(pos,mu,cov_matrix,d):        
	n = d
	powf=d/2
	#print(cov_matrix)
	#print(mu)
	Sigma_det = det(cov_matrix,d)
	#print(Sigma_det)
	inv_cov_matrix=inv(cov_matrix,d)
	N=Sigma_det
	#N = np.sqrt((2*np.pi)**n * Sigma_det)
	print("N=",N)

	# This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
	# way across all the input variables.
	fac = np.einsum('...k,kl,...l->...', pos-mu, inv_cov_matrix, pos-mu)
    
	#print(-fac/2)
	#if fac<=-100000:
	#	fac=-100000
	print("exp=",np.exp(-fac / 2))
	if np.exp(-fac / 2) < 0.00000000000001:
		print("verysmall")
	if 	np.exp(-fac / 2) > 100000000000000:
		print("verylarge")
	return np.exp(-fac / 2)/N




def Gmm(x,n,d,k,Z_nk,Pi_k,mean,cov_mat,cn,xg):
	print("cn=",cn)
	ans=np.zeros((d,d))
	diff=np.zeros(d)
	l_new=0.0
	l_old=0.00
	cy=0
	epoch=0
	while cy < 100 :
		cy=cy+1
		l_old=l_new
		l_new=0.0000
		total_sum=0.0
		for i in range(n):
			#print("i=",i)
			summ=0.0000;
			for j in range(k):
				#print(cov_mat[j])
				#print(G(x[i],mean[j],cov_mat[j],d))
				summ=summ+(Pi_k[j]*G(x[i],mean[j],cov_mat[j],d))
				#print(summ)
			for j in range(k):
				#print("sum is:",summ)
				Z_nk[i][j]=(Pi_k[j]*G(x[i],mean[j],cov_mat[j],d))*1.000/summ
			l_new += np.log(summ)
		if epoch < 100:
			xg[cn][epoch]=l_new
			#print(l_new)
			epoch+=1
		print("c=",cy)
		Pi_Z_nk_sum=np.zeros(k);
		mean_Z_nk_sum=np.zeros((k,d));
		cov_mat_Z_nk_sum=np.zeros((k,d,d))
		for i in range(n):
			for j in range(k):
				Pi_Z_nk_sum[j]+=Z_nk[i][j];
				mean_Z_nk_sum[j]=np.add(mean_Z_nk_sum[j],Z_nk[i][j]*x[i]);
				#cov_mat_Z_nk_sum[j]=np.add(cov_mat_Z_nk_sum[j],Z_nk[i][j]*(np.matmul(np.transpose(np.subtract(x[i],mean[j])),np.subtract(x[i],mean[j]))))
		
		for j in range(k):
			Pi_k[j]=(Pi_Z_nk_sum[j]/n)
			mean[j]=mean_Z_nk_sum[j]/Pi_Z_nk_sum[j]
			#cov_mat_Z_nk_sum[j]=cov_mat_Z_nk_sum[j]/Pi_Z_nk_sum[j];
		for i in range(n):
			for j in range(k):
				diff[:]=x[i]-mean[j]
				find_cov.one_d_matmul(diff,ans,d)
				cov_mat_Z_nk_sum[j]=np.add(cov_mat_Z_nk_sum[j],Z_nk[i][j]*ans)

        
		for j in range(k):
			if Pi_Z_nk_sum[j]!=0:
				cov_mat[j]=cov_mat_Z_nk_sum[j]/Pi_Z_nk_sum[j];

		#print(cov_mat)
		for r in range(d):
			for c in range(d):
				if r!=c:
					cov_mat[j][r][c]=0;
				else:
					if cov_mat[j][r][c]<=0.0001:
						cov_mat[j][r][c]=1.000				
		#print("mean",mean)
		print(l_old,l_new)
	    
	#print("mean..",mean)
	#print("cov:",cov_mat)

	xx=np.zeros(n)
	yy=np.zeros(n)
	#print(x)
	for i in range(n):
		xx[i]=x[i][0]
		yy[i]=x[i][1]
	#plt.plot(xx,yy,col,label=name,linestyle='None',marker='.')
		
