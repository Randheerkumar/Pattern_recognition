
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import bays_classifier
import gmm_cluster

def prob(pos,mu,cov_matrix,d):        
	n = d
	gmm_cluster.change_covmat(cov_matrix,d)
	inv_cov_matrix=np.linalg.inv(cov_matrix)
	Sigma_det = np.linalg.det(cov_matrix)
	N = np.sqrt((2*np.pi)**n * Sigma_det)

	# This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
	# way across all the input variables.
	fac = np.einsum('...k,kl,...l->...', pos-mu, inv_cov_matrix, pos-mu)

	return np.exp(-fac / 2) / N

def decplot(minx,maxx,miny,maxy,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d):
	
	#x=np.arange(minx-3,maxx+3,0.01)
	#=np.arange(miny-3,maxy+3,0.01)
	xn=np.zeros(2)
	ii=0;jj=0;kk=0
	#c=x.size
	#r=y.size
	#size=(maxx-minx+4)*(maxy-miny+4);
	#inc=(size*1.0)/5000;
	inc_x=((maxx-minx)*1.0)/400
	inc_y=((maxy-miny)*1.0)/400
	size=500000
	x1=np.zeros(size);
	y1=np.zeros(size);
	x2=np.zeros(size);
	y2=np.zeros(size);
	x3=np.zeros(size);
	y3=np.zeros(size);
	m=0.0;n=0.0;
	m=minx-2
	#inc_x=0.05
	#inc_y=0.05

	while m < maxx:
		n=miny-2
		while n < maxy:
			xn[0]=m
			xn[1]=n
			temp=np.zeros(3)
			for j in range(k):
				temp[0] += (Pi_k1[j]*prob(xn,mean1[j],cov_mat1[j],d))
			for j in range(k):
				temp[1] += (Pi_k2[j]*prob(xn,mean2[j],cov_mat2[j],d))
			for j in range(k):
				temp[2] += (Pi_k3[j]*prob(xn,mean3[j],cov_mat3[j],d))

			index=np.argmax(temp)
			if index==0:
				x1[ii]=xn[0]
				y1[ii]=xn[1]
				ii +=1
			     #plt.plot(i,j,'#DEB887')
			     #print("index=",index)
			elif index==1:
				x2[jj]=xn[0]
				y2[jj]=xn[1]
				jj +=1
				#plt.plot(i,j,'#53868B')
				#print("index=",index)
			elif index==2:	
				x3[kk]=xn[0]
				y3[kk]=xn[1]
				kk +=1
			n +=inc_y	
		m +=inc_x

	x11=np.zeros(ii)
	y11=np.zeros(ii)
	x22=np.zeros(jj)
	y22=np.zeros(jj)
	x33=np.zeros(kk)
	y33=np.zeros(kk)
	i=0
	for i in range(ii):
		x11[i]=x1[i]
		y11[i]=y1[i]
	i=0	
	for i in range(jj):
		x22[i]=x2[i]
		y22[i]=y2[i]
	i=0	
	for i in range(kk):
		x33[i]=x3[i]
		y33[i]=y3[i]


	plt.plot(x11,y11,'#DEB887',label='class1_predicted')
	plt.plot(x22,y22,'#53868B',label='class2_predicted')
	plt.plot(x33,y33,'#458B00',label='class3_predicted')
#plt.show()




def class_f(x,n,mean1,cov1,Pi_k1,mean2,cov2,Pi_k2,mean3,cov3,Pi_k3,k,d,cn,conf):
	#decplot(mean1,cov1,Pi_k1,mean2,cov2,Pi_k2,mean3,cov3,Pi_k3,k,d)
	# nt=int(raw_input("enter number of test clases :"))	
		
		for i in range(n):
			temp=np.zeros(3)
			for j in range(k):
				temp[0] += (Pi_k1[j]*prob(x[i],mean1[j],cov1[j],d))
			for j in range(k):
				temp[1] += (Pi_k2[j]*prob(x[i],mean2[j],cov2[j],d))
			for j in range(k):
				temp[2] += (Pi_k3[j]*prob(x[i],mean3[j],cov3[j],d))


			index=np.argmax(temp)
			conf[cn][index]+=1;
			'''
			if index==0:
				conf[cn][index]
				#plt.plot(x[i][0],x[i][1],'ro')
			elif index==1:
				index=1
				#plt.plot(x[i][0],x[i][1],'bo')
			elif index==2:
				index=2
				#plt.plot(x[i][0],x[i][1],'yo')	
			'''	
										
	#plt.show()
	#plt.legend()




