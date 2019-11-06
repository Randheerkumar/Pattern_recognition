import matplotlib.pyplot as plt
import numpy as np
import find_cov


def G(pos,mu,cov_matrix,d):        
	n = d
	inv_cov_matrix=np.linalg.inv(cov_matrix)
	Sigma_det = np.linalg.det(cov_matrix)
	N = np.sqrt((2*np.pi)**n * Sigma_det)

	# This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
	# way across all the input variables.
	fac = np.einsum('...k,kl,...l->...', pos-mu, inv_cov_matrix, pos-mu)

	return np.exp(-fac / 2) / N


def plot_gmm(Z_nk,mean,cov_mat,x,k,n):
	n1=0;n2=0;n3=0;
	for i in range(n):
		index=np.argmax(Z_nk[i])
		if index==0:
			n1+=1;
		elif index==1:
			n2+=1;
		elif index==2:
			n3+=1;

	print("n3=",n3)			
	x1=np.zeros((2,n1));
	x2=np.zeros((2,n2));
	x3=np.zeros((2,n3));
	n1=0;n2=0;n3=0;
	for i in range(n):
		index=np.argmax(Z_nk[i])
		if index==0:
			x1[0][n1]=x[i][0];
			x1[1][n1]=x[i][1];
			n1+=1;
		elif index==1:
			x2[0][n2]=x[i][0];
			x2[1][n2]=x[i][1];
			n2+=1;
		elif index==2:
			x3[0][n3]=x[i][0];
			x3[1][n3]=x[i][1];
			n3+=1;	

	#print("plot k-means stars")
	print("n1=",n1,"n2=",n2,"n3=",n3)
	plt.plot(x1[0],x1[1],'ro')	
	plt.plot(x2[0],x2[1],'bo')
	plt.plot(x3[0],x3[1],'go')		
	plt.plot(mean[0][0],mean[0][1],'y*')
	plt.plot(mean[1][0],mean[1][1],'y*')
	plt.plot(mean[2][0],mean[2][1],'y*')
	plt.show()
		

def Gmm(x,n,d,k,Z_nk,Pi_k,mean,cov_mat):
	ans=np.zeros((d,d))
	diff=np.zeros(d)
	l_new=0.0
	l_old=0.00
	c=0
	l_gy=np.zeros(n);
	l_g1=0;
	while c < 2 or l_new-l_old>0.001 :
		c=c+1
		l_old=l_new
		l_new=0.0000
		for i in range(n):
			summ=0.0000;
			for j in range(k):

				summ=summ+(Pi_k[j]*G(x[i],mean[j],cov_mat[j],d))
			for j in range(k):
				#print("sum is:",summ)
				Z_nk[i][j]=(Pi_k[j]*G(x[i],mean[j],cov_mat[j],d))*1.000/summ

			l_new+=np.log(summ);

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
		print(l_old,l_new)
	    #print("cov:",cov_mat)
		l_gy[l_g1]=l_new;
		#l_gx[l_g1]=l_g1+1;
		l_g1+=1

	
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
	print("mean..",mean)
	print("cov:",cov_mat)
	#plot_gmm(x,k,n,Z_nk,mean,cov_mat)
	plot_gmm(Z_nk,mean,cov_mat,x,k,n)



'''
x1=np.zeros(n);
y1=np.zeros(n);
x2=np.zeros(n);
y2=np.zeros(n);
x3=np.zeros(n);
y3=np.zeros(n);
n1=0;n2=0;n2=0;
for i in range(n):
	index=argmax(Z_nk[i]);
	if index==0:
		x1[n1]=x[i][0];
		y1[n1]=x[i][1]
		n1+=1
	elif index==1:
		x2[n2]=x[i][0]
		y2[n2]=x[i][1]
		n2+=1
	elif index==2:
		x3[n3]=x[i][0];
		y3[n3]=x[i][1]
		n3+=1			
xx=np.zeros(n)
yy=np.zeros(n)
for i in range(n):
	xx[i]=x[i][0]
	yy[i]=x[i][1]

'''
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
'''
col="#7FFFD4"
col1="red"
name="class1_data"
cn=0;
if cn==0:
	col="#7FFFD4"
	col1="red"
elif cn==1:
	col="#FFE4C4"
	col1="#000000"
	name="class2_data"
elif cn==2:
	col="#C1FFC1"
	col1="#0000FF"
	name="class3_data"

plt.plot(xx,yy,col,label=name,linestyle='None',marker='o')
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
'''
	














