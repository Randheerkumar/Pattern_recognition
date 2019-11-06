'''
Pattern recognition cs669

lab 2

Group10

'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt                          # for plotting
import find_cov
import k_means
import gmm_cluster
import bays_classifier
import temp


k=32                  #number of cluster

#training data
k=int(raw_input("enter number of cluster"))
c1="Real_data/r1.txt"
c2="Real_data/r2.txt"
c3="Real_data/r3.txt"


#test data
tc1="Real_data/rt1.txt"
tc2="Real_data/rt2.txt"
tc3="Real_data/rt3.txt"

xg=np.zeros((3,100))


def mainf(x,k,d,n,znk,Pi_k,mean,cov_mat,cn):
	znk=k_means.parameter(x,k,d,n,mean) #clustering using k-means

	find_cov.cov_mat_new(x,n,k,d,znk,Pi_k,mean,cov_mat) #intial parameter for gmm

	gmm_cluster.Gmm(x,n,d,k,znk,Pi_k,mean,cov_mat,cn,xg) #gmm clustering
	#print("...",x)



i=0
file1=open(c1,"r")
for line in file1:
		i=i+1;
file1.close()
n1=i

i=0
file1=open(c2,"r")
for line in file1:
		i=i+1;
file1.close()
n2=i

i=0
file1=open(c3,"r")
for line in file1:
		i=i+1;
file1.close()
n3=i
print("n1,n2,n3",n1,n2,n3)


d=2                       #dimension of a vector
cl=3                     #number of class



x1=np.zeros((n1,d))
mean1=np.zeros((k,d))
znk1=np.zeros((n1,k))
Pi_k1=np.zeros(k);
cov_mat1=np.zeros((k,d,d))


x2=np.zeros((n2,d))
mean2=np.zeros((k,d))
znk2=np.zeros((n2,k))
Pi_k2=np.zeros(k);
cov_mat2=np.zeros((k,d,d))

x3=np.zeros((n3,d))
mean3=np.zeros((k,d))
znk3=np.zeros((n3,k))
Pi_k3=np.zeros(k);
cov_mat3=np.zeros((k,d,d))

x11=np.zeros(n1)
x22=np.zeros(n2)
x33=np.zeros(n3)
y11=np.zeros(n1)
y22=np.zeros(n2)
y33=np.zeros(n3)


minx=5000.0
miny=5000.0
maxx=-5000.0
maxy=-5000.0

i=0
file=open(c1,"r")
for line in file:
	a=line.split()
	for j in range(len(a)):
		x1[i][j]=float(a[j])
	if minx > x1[i][0]:
		minx=x1[i][0]
	if maxx < x1[i][0]:
		maxx=x1[i][0]	
	if miny > x1[i][1]:
		miny=x1[i][1]
	if maxy < x1[i][1]:
		maxy=x1[i][1]		
	x11[i]=x1[i][0]
	y11[i]=x1[i][1]
	i=i+1
file.close()
#print(x1)
i=0
file=open(c2,"r")
for line in file:
	a=line.split()
	for j in range(len(a)):
		x2[i][j]=float(a[j])
	if minx > x2[i][0]:
		minx=x2[i][0]
	if maxx < x2[i][0]:
		maxx=x2[i][0]	
	if miny > x2[i][1]:
		miny=x2[i][1]
	if maxy < x2[i][1]:
		maxy=x2[i][1]
	x22[i]=x2[i][0]
	y22[i]=x2[i][1]
	i=i+1
i=0
file.close()
file=open(c3,"r")
for line in file:
	a=line.split()
	for j in range(len(a)):
		x3[i][j]=float(a[j])
	if minx > x3[i][0]:
		minx=x3[i][0]
	if maxx < x3[i][0]:
		maxx=x3[i][0]	
	if miny > x3[i][1]:
		miny=x3[i][1]
	if maxy < x3[i][1]:
		maxy=x3[i][1]	
	x33[i]=x3[i][0]
	y33[i]=x3[i][1]
	i=i+1
file.close()	

print(minx,maxx,miny,maxy)

mainf(x1,k,d,n1,znk1,Pi_k1,mean1,cov_mat1,0)		
mainf(x2,k,d,n2,znk2,Pi_k2,mean2,cov_mat2,1)
mainf(x3,k,d,n3,znk3,Pi_k3,mean3,cov_mat3,2)
print(cov_mat1)
print(mean1)
print(cov_mat2)
print(mean2)
print(cov_mat3)
print(mean3)
plt.legend()
plt.show()

i=0
file1=open(tc1,"r")
for line in file1:
		i=i+1;
file1.close()
tn1=i

i=0
file1=open(tc2,"r")
for line in file1:
		i=i+1;
file1.close()
tn2=i

i=0
file1=open(tc3,"r")
for line in file1:
		i=i+1;
file1.close()
tn3=i
print("n1,n2,n3",tn1,tn2,tn3)
tx1=np.zeros((n1,d))
tx2=np.zeros((n2,d))
tx3=np.zeros((n3,d))

i=0
file=open(tc1,"r")
for line in file:
	a=line.split()
	for j in range(len(a)):
		tx1[i][j]=float(a[j])
	i=i+1

i=0
file=open(tc2,"r")
for line in file:
	a=line.split()
	for j in range(len(a)):
		tx2[i][j]=float(a[j])
	i=i+1
i=0
file=open(tc3,"r")
for line in file:
	a=line.split()
	for j in range(len(a)):
		tx3[i][j]=float(a[j])
	i=i+1





confusion_m=np.zeros((cl,cl))
bays_classifier.class_f(tx1,tn1,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d,0,confusion_m)
#print(confusion_m)
bays_classifier.class_f(tx2,tn2,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d,1,confusion_m)
#print(confusion_m)
bays_classifier.class_f(tx3,tn3,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d,2,confusion_m)
print(k)
print(confusion_m)
'''
print(cov_mat1)
print(cov_mat2)
print(cov_mat3)
'''
temp.decplot(minx,maxx,miny,maxy,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d)

#bays_classifier.decplot(minx,maxx,miny,maxy,mean1,cov_mat1,Pi_k1,mean2,cov_mat2,Pi_k2,mean3,cov_mat3,Pi_k3,k,d)
plt.plot(x11,y11,'ro',label='class1')
plt.plot(x22,y22,'bo',label='class2')
plt.plot(x33,y33,'go',label='class3')
plt.show()


#print(xg)
col=(['red','blue','green'])
name=(['class1','class2','class3'])

for j in range(0,3):
	count=0;
	for i in range(100):
			if xg[j][i]==0:
				break;
			count+=1;
	temp=np.zeros(count)
	epoch=np.zeros(count)
	for i in range(0,count):
		temp[i]=xg[j][i]	
		epoch[i]=i+1
	plt.plot(epoch,temp,color=col[j],label=name[j],linestyle='-',marker='o')
plt.legend()
plt.show()
'''
fig=plt.figure()
fig,savefig('1.png')
'''		
				