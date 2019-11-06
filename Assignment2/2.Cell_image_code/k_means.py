import numpy as np
from tqdm  import tqdm
import matplotlib.pyplot as plt         # for plotting

def dot_prod(d,x):
	ans=0.00
	for i in range(d):
		ans+=(x[i]*x[i]);

	return(ans);	



def f_plot(znk,mean,x,k,n):
	n1=0;n2=0;n3=0;
	for i in range(n):
		if znk[i][0]==1:
			n1+=1;
		elif znk[i][1]==1:
			n2+=1;
		elif znk[i][2]==1:
			n3+=1;
	x1=np.zeros((2,n1));
	x2=np.zeros((2,n2));
	x3=np.zeros((2,n3));
	n1=0;n2=0;n3=0;
	for i in range(n):
		if znk[i][0]==1:
			x1[0][n1]=x[i][0];
			x1[1][n1]=x[i][1];
			n1+=1;
		elif znk[i][1]==1:
			x2[0][n2]=x[i][0];
			x2[1][n2]=x[i][1];
			n2+=1;
		elif znk[i][2]==1:
			x3[0][n3]=x[i][0];
			x3[1][n3]=x[i][1];
			n3+=1;	

	print("plot k-means stars")
	print("n1=",n1,"n2=",n2,"n3=",n3)
	plt.plot(x1[0],x1[1],'ro')	
	plt.plot(x2[0],x2[1],'bo')
	plt.plot(x3[0],x3[1],'go')		
	plt.plot(mean[0][0],mean[0][1],'y*')
	plt.plot(mean[1][0],mean[1][1],'y*')
	plt.plot(mean[2][0],mean[2][1],'y*')
	plt.show()
		


def mean_and_cost(mean,x,znk,k,size,d):
		temp_array=np.zeros(d)
		new=0.0
		for j in range(0,k):    #j represents jth cluster 
			nr=np.zeros(d)      #numerator term          
			dr=0.0000000        #denominator term
			for i in range(0,size):     # f is the fth feature of a vector
				for f in range(0,d):   
					nr[f] += znk[i][j]*x[i][f]
				dr += znk[i][j]	
			if(dr > 0):
				mean[j]=nr/dr
					#mean_y[j]=nry/dr
				#else :
				 #   mean[j][f]=0.0
				    #mean_y[j]=0.0
		print("mean_and_cost")
		for j in range(0,k):
			for i in range(0,size) :
				temp_array[:]=np.subtract(mean[j],x[i])
				new += znk[i][j]*(np.matmul(temp_array,np.transpose(temp_array)))
                
	
		#print(new,new)
		return np.log(new)


def parameter(x,k,d,n,mean) :
	temp_array=np.zeros(d)
	
	for i in range(0,k):
		for f in range(0,d) :
			mean[i][f]=x[i][f]                                #randomly choosing mean for all cluster
   
	#mean[0][0]=228.87452152;mean[0][1]=4.59432967;
	#mean[1][0]=197.065907;mean[1][1]=120.32030662;
	#mean[2][0]=219.94223947;mean[2][1]=471.08539217;

	temp=np.zeros(k)
	temp_array=np.zeros((k,d))
	l_gy=np.zeros(n);

	l_g1=0;		                              
	old=0.0
	new=0.0
	c=0
	while c < 2 or abs(old-new) > 0.001 :
		c=c+1
		old=new
		znk=np.zeros((n,k))
		new=0.0
		#print("c is",c)
		for i in range(0,n):
			for j in range(0,k):
				temp_array[j]=np.subtract(mean[j],x[i])
				#temp[j]=np.matmul(temp_array,np.transpose(temp_array))
				temp[j]=dot_prod(d,temp_array[j])
			index=np.argmin(temp)
			znk[i][index]=1		
		new=mean_and_cost(mean,x,znk,k,n,d)
	
		l_gy[l_g1]=new;
		#l_gx[l_g1]=l_g1+1;
		l_g1+=1
		#plt.plot(l_g1,new,color='cornflowerblue',linestyle='-',marker='o')
		#plt.plot(l_g1,new)
		#f_plot(znk,mean,x,k,n)
		print("new is",new,"old=",old)
		print("in loop for k means")
	
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
	f_plot(znk,mean,x,k,n)

	return(znk)	

   