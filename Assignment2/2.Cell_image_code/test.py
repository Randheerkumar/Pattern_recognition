#importing the required librabries
import imageio
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from PIL import Image

def G(pos,mu,cov_matrix,d):        
	n = d
	inv_cov_matrix=np.linalg.inv(cov_matrix)
	Sigma_det = np.linalg.det(cov_matrix)
	N = np.sqrt((2*np.pi)**n * Sigma_det)

	# This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
	# way across all the input variables.
	fac = np.einsum('...k,kl,...l->...', pos-mu, inv_cov_matrix, pos-mu)

	return np.exp(-fac / 2) / N


'''function 'extract_feature' take three parameters pic and file and patch_size .pic is the one dimensional pexels value a cell image and
file is the a text file where the mean and variance of 7*7 patch of pixels  to be written'''

def test_img(pic,file,patch_size,mean):
	height=pic.shape[0]
	width=pic.shape[1]
	h=height%patch_size   
	w=width%patch_size
	cond=0;cond1=0
	x1=np.zeros((2,height*height))
	x2=np.zeros((2,height*height))
	x3=np.zeros((2,height*height))
	n1=0;n2=0;n3=0;
	#patch=queue.Queue(maxsize=patch_size*patch_size)

	temp=np.zeros(())
	temp_array=np.zeros(2)

	i=0;j=0;
	f=open(file,"a")                                                  #open the file where mean and variance pair are to be stacked
	while i<height-h:
		j=0
		while j< width-w:
			r=i;mean1=0.000;sum1=0;sq_sum1=0;var1=0.000;k=0;
			temp=np.zeros((2,patch_size*patch_size))
			while r<i+patch_size:
				c=j
				while c<j+patch_size:
					sum1 += int(pic[r][c])                             #sum of pixels in 7*7 patch
					#print(int(pic[r][c]))
					sq_sum1 += (int(pic[r][c])*int(pic[r][c]))         #square sum of the pixels in 7*7 patch        
					#ans=(int(pic[r][c])*int(pic[r][c]))
					#print(pic[r][c],ans)
					temp[0][k]=r;temp[1][k]=c;
					k+=1
					c+=1
				r+=1
            

			x=np.zeros(2)
			temp1=np.zeros(3)
			mean1=(sum1*1.00000)/(patch_size*patch_size)              # calulating maen of pixels values in each 7*7 overlapping patch
			#print(sq_sum1,mean1)
			#var1=0;
			#print(sum1,sq_sum1)
			var1=((sq_sum1*1.00000)/(patch_size*patch_size))-(mean1*mean1)  # calulating variance of pixels values in each 7*7 overlapping patch
			#print(mean1,var1)
			x[0]=mean1;
			x[1]=var1;
			for jn in range(0,3):
				temp_array[:]=np.subtract(mean[jn],x)
				temp1[jn]=np.matmul(temp_array,temp_array)
			index=np.argmin(temp1)
			if index==0:
				for jn in range(49):
					x1[0][n1]=temp[0][jn]
					x1[1][n1]=temp[1][jn]
					n1+=1;

			if index==1:
				for jn in range(49):
					x2[0][n2]=temp[0][jn]
					x2[1][n2]=temp[1][jn]
					n2+=1;

			if index==2:
				for jn in range(49):
					x3[0][n3]=temp[0][jn]
					x3[1][n3]=temp[1][jn]
					n3+=1;		
						


			j+=patch_size
			#if j>width-w-1 and cond1==0 and width%patch_size!=0:                               #this loop ensures that no pixels is left even if there is not 32*32 patch we make we take some pixels from prev patch make 32*32 patch                           
			#	j=width-patch_size
			#	cond1=1	
		i+=patch_size
		#cond1=0
		#if i>height-h-1 and cond==0 and height%patch_size!=0:
		#	i=height-patch_size
		#	cond=1
		#	cond1=0	
	x11=np.zeros((2,n1));
	x22=np.zeros((2,n2));
	x33=np.zeros((2,n3));
	i=0;
	print("cluster1 starts")
	for i in range(n1):
		x11[0][i]=x1[0][i];
		x11[1][i]=x1[1][i]
		print(i,x1[0][i],x1[1][0])

	print("cluster2 starts")
	for i in range(n2):
		x22[0][i]=x2[0][i];
		x22[1][i]=x2[1][i]
		#print(i,x2[0][i],x2[1][0])


	print("cluster3 starts")

	for i in range(n3):
		x33[0][i]=x3[0][i];
		x33[1][i]=x3[1][i]		
		print(i,x3[0][i],x3[1][i])

	print("n1=",n1,"n2=",n2,"n3=",n3)
	plt.plot(x11[0],x11[1],'ro')
	plt.plot(x22[0],x22[1],'bo')
	plt.plot(x33[0],x33[1],'go')
	plt.show();
	f.close()


def test_img_gmm(pi_k,mean,cov_mat,pic,patch_size,d):
	height=pic.shape[0]
	width=pic.shape[1]
	h=height%patch_size   
	w=width%patch_size
	cond=0;cond1=0;
	x1=np.zeros((2,height*height))
	x2=np.zeros((2,height*height))
	x3=np.zeros((2,height*height))
	n1=0;n2=0;n3=0;
	#patch=queue.Queue(maxsize=patch_size*patch_size)

	temp=np.zeros(())
	temp_array=np.zeros(2)

	i=0;j=0;                                                 #open the file where mean and variance pair are to be stacked
	while i<height-h:
		j=0
		while j< width-w:
			r=i;mean1=0.000;sum1=0;sq_sum1=0;var1=0.000;k=0;
			temp=np.zeros((2,patch_size*patch_size))
			while r<i+patch_size:
				c=j
				while c<j+patch_size:
					sum1 += int(pic[r][c])                             #sum of pixels in 7*7 patch
					#print(int(pic[r][c]))
					sq_sum1 += (int(pic[r][c])*int(pic[r][c]))         #square sum of the pixels in 7*7 patch        
					#ans=(int(pic[r][c])*int(pic[r][c]))
					#print(pic[r][c],ans)
					temp[0][k]=r;temp[1][k]=c;
					k+=1
					c+=1
				r+=1
            

			x=np.zeros(2)
			temp1=np.zeros(3)
			mean1=(sum1*1.00000)/(patch_size*patch_size)              # calulating maen of pixels values in each 7*7 overlapping patch
			#print(sq_sum1,mean1)
			#var1=0;
			#print(sum1,sq_sum1)
			var1=((sq_sum1*1.00000)/(patch_size*patch_size))-(mean1*mean1)  # calulating variance of pixels values in each 7*7 overlapping patch
			#print(mean1,var1)
			x[0]=mean1;
			x[1]=var1;
			summ=0;
			for jn in range(0,3):
				summ=summ+(pi_k[jn]*G(x,mean[jn],cov_mat[jn],d))
             
			for jn in range(0,3):
				temp1[jn]=(pi_k[jn]*G(x,mean[jn],cov_mat[jn],d))/summ;

			index=np.argmax(temp1)
			if index==0:
				for jn in range(49):
					x1[0][n1]=temp[0][jn]
					x1[1][n1]=temp[1][jn]
					n1+=1;

			if index==1:
				for jn in range(49):
					x2[0][n2]=temp[0][jn]
					x2[1][n2]=temp[1][jn]
					n2+=1;

			if index==2:
				for jn in range(49):
					x3[0][n3]=temp[0][jn]
					x3[1][n3]=temp[1][jn]
					n3+=1;		
						


			j+=patch_size
			#if j>width-w-1 and cond1==0 and width%patch_size!=0:                               #this loop ensures that no pixels is left even if there is not 32*32 patch we make we take some pixels from prev patch make 32*32 patch                           
			#	j=width-patch_size
			#	cond1=1	
		i+=patch_size
		#cond1=0
		#if i>height-h-1 and cond==0 and height%patch_size!=0:
		#	i=height-patch_size
		#	cond=1
		#	cond1=0	
	x11=np.zeros((2,n1));
	x22=np.zeros((2,n2));
	x33=np.zeros((2,n3));
	i=0;
	for i in range(n1):
		x11[0][i]=x1[0][i];
		x11[1][i]=x1[1][i]
		#print(x1[0][i],x1[1][0])

	for i in range(n2):
		x22[0][i]=x2[0][i];
		x22[1][i]=x2[1][i]
		#print(x1[0][i],x1[1][0])

	for i in range(n3):
		x33[0][i]=x3[0][i];
		x33[1][i]=x3[1][i]		
        print(x1[0][i],x1[1][0])

	print("n1=",n1,"n2=",n2,"n3=",n3)
	plt.plot(x11[0],x11[1],'ro')
	plt.plot(x22[0],x22[1],'bo')
	plt.plot(x33[0],x33[1],'go')
	plt.show();	