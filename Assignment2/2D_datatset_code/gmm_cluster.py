import matplotlib.pyplot as plt
import numpy as np
import find_cov

def change_covmat(cov_matrix,d):
	for i in range(d):
		for j in range(d):
			if i==j :
				if cov_matrix[i][i]==0:
					cov_matrix[i][i]=1
		else:
			cov_matrix[i][i]=np.power(cov_matrix[i][i],(1.0/3))	

def G(pos,mu,cov_matrix,d):        
	n = d
	#print(cov_matrix)
	change_covmat(cov_matrix,d)
	#print("G fn",cov_matrix)
	inv_cov_matrix=np.linalg.inv(cov_matrix)
	Sigma_det = np.linalg.det(cov_matrix)
	N = np.sqrt((2*np.pi)**n * Sigma_det)

	# This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
	# way across all the input variables.
	fac = np.einsum('...k,kl,...l->...', pos-mu, inv_cov_matrix, pos-mu)
	fac=np.power(fac,(1.0/3))
	return np.exp(-fac / 2) / N



def Gmm(x,n,d,k,Z_nk,Pi_k,mean,cov_mat,cn,xg):
	print("cn=",cn)
	ans=np.zeros((d,d))
	diff=np.zeros(d)
	l_new=0.0
	l_old=0.00
	c=0
	epoch=0
	while c < 2 or l_new-l_old>0.001 :
		c=c+1
		l_old=l_new
		l_new=0.0000
		total_sum=0.0
		for i in range(n):
			summ=0.0000;
			for j in range(k):
				summ=summ+(Pi_k[j]*G(x[i],mean[j],cov_mat[j],d))
			for j in range(k):
				#print("sum is:",summ)
				Z_nk[i][j]=(Pi_k[j]*G(x[i],mean[j],cov_mat[j],d))*1.000/summ
			l_new += np.log(summ)
		#print(l_new)	
		if epoch < 100:
			xg[cn][epoch]=l_new
			#print(l_new)
			epoch+=1

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
			cov_mat[j]=cov_mat_Z_nk_sum[j]/Pi_Z_nk_sum[j];
		#print("mean",mean)
	    #print("cov:",cov_mat)
	    
	#print("mean..",mean)
	#print("cov:",cov_mat)

	xx=np.zeros(n)
	yy=np.zeros(n)
	#print(x)
	for i in range(n):
		xx[i]=x[i][0]
		yy[i]=x[i][1]


	'''
		for j in range(n):


			if np.argmax(Z_nk[i])==0:	
				plt.plot(x[i][0],x[i][1],'ro')
			elif np.argmax(Z_nk[i])==1:
				plt.plot(x[i][0],x[i][1],'bo')
			elif np.argmax(Z_nk[i])==2:
				plt.plot(x[i][0],x[i][1],'go')
			elif np.argmax(Z_nk[i])==3:
				plt.plot(x[i][0],x[i][1],'yo')
    
	'''
	col="red"
	col1="#A52A2A"
	name="class1_data"
	if cn==0:
		col="red"
		col1="#A52A2A"
	elif cn==1:
		col="blue"
		col1="#6495ED"
		name="class2_data"
	elif cn==2:
		col="green"
		col1="#228B22"
		name="class3_data"

	plt.plot(xx,yy,col,label=name,linestyle='None',marker='.')
	for j in range(k):
		N=60
		min_x=xx[np.argmin(xx)]
		min_x=min_x-(min_x*5)/100

		max_x=xx[np.argmax(xx)]
		max_x=max_x+(max_x*5)/100

		min_y=yy[np.argmin(yy)]
		max_y=yy[np.argmax(yy)]

		min_y=min_y-(min_y*5)/100
		max_y=max_y+(max_y*5)/100

		X=np.linspace(min_x,max_x,N)   #x[],y[]
		Y=np.linspace(min_y,max_y,N)
		X,Y=np.meshgrid(X,Y)
		pos=np.empty(X.shape+(2,))
		pos[:,:,0]=X
		pos[:,:,1]=Y
		Z=G(pos,mean[j],cov_mat[j],d)
		plt.contour(X,Y,Z,colors=col1)

		

	












