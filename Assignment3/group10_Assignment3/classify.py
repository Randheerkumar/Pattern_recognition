#importing the required Libararies
import numpy as np
import glob

#******************************************************************************************************************************
#this function assigned the initial state sequence of an observation
def find_initial_state_seq(x,l,N):
	interval=l/N;
	state=0;
	i=0;
	while i<l-(l%N):
		for j in range(interval):
			x[i]=state
			i+=1
		state+=1
	while(i<l):
		x[i]=state-1;
		i+=1		

#*********************************************************************************************************************************
# this function finds the state transition probality matrix given the observation sequence and state sequence
def find_A(A,O,state_seq,T,N):
	count_A=np.zeros(N);
	for i in range(T-1):
		A[int(state_seq[i])][int(state_seq[i+1])]+=1;
		#print(state_seq[i],state_seq[i+1]);
		count_A[int(state_seq[i])]+=1;
		#print(int(state_seq[i]));

	#count_A[int(state_seq[T-1])]+=1;
	#print(A);
	for i in range(N):
		for j in range(N):
			A[i][j]=(A[i][j]*1.000)/count_A[i];

#****************************************************************************************************************************			
#this function find the state observation probability
def find_B(B,O,state_seq,T,N,M):
	count_B=np.zeros(N);
	for i in range(T):
		count_B[int(state_seq[i])]+=1;
		B[int(state_seq[i])][int(O[i])]+=1;

	for i in range(N):
		for j in range(M+1):
			B[i][j]=(B[i][j]*1.00)/count_B[i];

	
#**************************************************************************************************************************
#this function calculate initial state probabilty given Gamma
def find_Pi11(Pi11,Gamma,N):
	for i in range(N):
		Pi11[i]=Gamma[0][i];

#**************************************************************************************************************************		
#this function calculate A given Zeta and Gamma                           
def find_A11(A11,Zeta,Gamma,T,M,N):
	for i in range(N):
		sum_Gamma=0.0;
		for t in range(T-1):
			sum_Gamma+=Gamma[t][i];
		for j in range(N):
			sum_Zeta=0.0;
			for t in range(T-1):
				sum_Zeta+=Zeta[t][i][j];
			A11[i][j]=(sum_Zeta*1.00)/sum_Gamma;	

#***************************************************************************************************************************
#this function calculate B given Zeta and Gamma
def find_B11(B11,Zeta,Gamma,T,N,M,O):
	for j in range(N):
		dr_sum=0;
		for t in range(T):
			dr_sum+=Gamma[t][j];
		k=1;
		while k<M+1:
			nr_sum=0.0;
			for t in range(T):
				if(int(O[t])==k):
					nr_sum+=Gamma[t][j];
		
			B11[j][k]=(nr_sum*1.00)/dr_sum;	
			k+=1;	



#******************************************************************************************************************************
#this function find Alpha given the observation sequence and Lamda i.e A,B and Pi
def find_Alpha(A,B,Pi,O,N,M,T,Alpha):
	for j in range(N):
		Alpha[0][j]=Pi[j]*B[j][int(O[0])];

	t=1;
	while t<T:
		for j in range(N):
			summ=0.0;
			for i in range(N):
				summ+=(Alpha[t-1][i]*A[i][j]);
			summ*=B[j][int(O[t])];
			Alpha[t][j]=summ;
		t+=1;
    

#*******************************************************************************************************************************				
#this function find Beta given the observation sequence and Lamda i.e A,B and Pi
def find_Beta(A,B,Pi,O,N,M,T,Beta):
	for j in range(N):
		Beta[T-1][j]=1;

	t=T-2;
	while t>0:
		for i in range(N):
			summ=0.00;
			for j in range(N):
				summ+=A[i][j]*B[j][int(O[t+1])]*Beta[t+1][j];
			Beta[t][i]=summ;
		t-=1;

	for i in range(N):
		summ=0.00;
		for j in range(N):
			summ+=A[i][j]*B[j][int(O[1])]*Beta[1][j];
		Beta[0][i]=summ;			

#**********************************************************************************************************************************
#this function calculate Zeta gieven A,B,Alpha,Beta
def find_Zeta(Alpha,Beta,A,B,Zeta,O,T,N,M):
	for t in range(T-1):
		total=0.00;
		for i in range(N):
			for j in range(N):
				total+=Alpha[t][i]*A[i][j]*B[j][int(O[t+1])]*Beta[t+1][j];
		for i in range(N):
			for j in range(N):
				Zeta[t][i][j]=(Alpha[t][i]*A[i][j]*B[j][int(O[t+1])]*Beta[t+1][j])/total;

#**********************************************************************************************************************************				
#this fucnction calculate Gamma given Alpha and Beta
def find_Gamma(Alpha,Beta,Gamma,T,N):
	for t in range(T):
		total=0.00;
		for i in range(N):
			total+=Alpha[t][i]*Beta[t][i];
		for i in range(N):
			Gamma[t][i]+=(Alpha[t][i]*Beta[t][i])/total;



#**************************************************************************************************************************
def Initial_parameter(M,N,A,B,Pi,source_path1):
	file=open(source_path1,"r");                                        
	i=0;
	print("hiiii")
	for line in file:                                                                       # length of observation                                                               
		a=line.split();
		lenn=len(a);
		T=lenn;   
		O=np.zeros(lenn) 
		for j in range(len(a)):
			O[j]=int(a[j]);
		x_state_seq=np.zeros(lenn)
		find_initial_state_seq(x_state_seq,lenn,N);
		#print(O)
		#print(x_state_seq)
		a=np.zeros((N,N));
		find_A(a,O,x_state_seq,T,N)
		#print(a);

		A=np.add(A,a);
		b=np.zeros((N,M+1));	
		find_B(b,O,x_state_seq,T,N,M);
		B=np.add(B,b);
		#print(b);
		#break;
		i+=1
		#if i>80:
		#	break;

	A=A/i;
	B=B/i;
	return A,B;

	
#*************************************************************************************************************************
def find_Lamda(M,N,A,B,Pi,source_path1):                                                        # this for storing the state sequenec
	
	A,B=Initial_parameter(M,N,A,B,Pi,source_path1)

	new=1
	old=0.00
	c=0;
    
	while(c<3 or abs(new-old)>0.001):
		file=open(source_path1,"r");
		c+=1;

		i=0;old=new;new=0;
		A1=np.zeros((N,N));                                                                      #state transition probablity  matrix  A                                                                                                                                     
		B1=np.zeros((N,M+1));                                                                    # state observation probability matrixb                                                                                                                                                       
		Pi1=np.zeros(N);  
		for line in file:                                                                                                               
			a=line.split();
			lenn=len(a);
			O=np.zeros(len(a)) 
			for j in range(len(a)):
				O[j]=int(a[j]);

			T=lenn;                                                                             
			Zeta=np.zeros((lenn,N,N))
			Gamma=np.zeros((lenn,N));
			Alpha=np.zeros((lenn,N));
			Beta=np.zeros((lenn,N));
			find_Alpha(A,B,Pi,O,N,M,T,Alpha);
			find_Beta(A,B,Pi,O,N,M,T,Beta);
			#print(O);
			#print(Alpha);
			#print(Beta);
			#return A,B,Pi;

			find_Zeta(Alpha,Beta,A,B,Zeta,O,T,N,M);
			#print(Zeta);
			find_Gamma(Alpha,Beta,Gamma,T,N);
			#print(Gamma)
			A11=np.zeros((N,N));                                                                      #state transition probablity  matrix  A                                                                                                                                   
			B11=np.zeros((N,M+1));                                                                    # state observation probability matrixb                                                                                                                                                     
			Pi11=np.zeros(N);
			find_Pi11(Pi11,Gamma,N)
			#print(Pi11) 
			find_A11(A11,Zeta,Gamma,T,M,N);
			find_B11(B11,Zeta,Gamma,T,N,M,O);
			Pi1=np.add(Pi1,Pi11);
			A1=np.add(A1,A11);
			#print(A11);
			B1=np.add(B1,B11);
			#print(B11)
			i+=1;
			neww=0;
			for j in range(N):
				neww+=Alpha[T-1][j];
				#print(Alpha[T-1][j]);
			#print(neww);	
			new=new+np.log(neww);
			#if i>80:
			#	break;

		A=A1/i;
		Pi=Pi1/i;
		B=B1/i;
		#new=new;
		#print(A);
		#break;
		#file.close();
		print(old,new);
		#print(old,new);
	return A,B,Pi;

#********************************************************************************************************************************

def Bayes_classifier(confusion_matrix,cl,A1,B1,Pi1,A2,B2,Pi2,A3,B3,Pi3,N,M,source_path1t):
	file=open(source_path1t,"r");
	i=0;
	for line in (file):
		a=line.split();
		T=len(a);   
		O=np.zeros(T) 
		for j in range(len(a)):
			O[j]=int(a[j]);

		Alpha1=np.zeros((T,N));
		Alpha2=np.zeros((T,N));
		Alpha3=np.zeros((T,N));
		find_Alpha(A1,B1,Pi1,O,N,M,T,Alpha1);
		find_Alpha(A2,B2,Pi2,O,N,M,T,Alpha2);
		find_Alpha(A3,B3,Pi3,O,N,M,T,Alpha3);
		#	print(Alpha1)
		temp=np.zeros(3);
		for j in range(N):
			temp[0]+=Alpha1[T-1][j];
		for j in range(N):
			temp[1]+=Alpha2[T-1][j];
		for j in range(N):
			temp[2]+=Alpha3[T-1][j];
        
		#print(temp[0],temp[1],temp[2]);
		index=np.argmax(temp);
		confusion_matrix[cl][index]+=1;
		i+=1	
	file.close();			

	print(confusion_matrix);
