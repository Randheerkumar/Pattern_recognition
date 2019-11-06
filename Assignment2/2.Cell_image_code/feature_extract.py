#importing the required librabries
import imageio
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from PIL import Image
#import queue



'''function 'extract_feature' take three parameters pic and file and patch_size .pic is the one dimensional pexels value a cell image and
file is the a text file where the mean and variance of 7*7 patch of pixels  to be written'''

def extract_feature(pic,file,patch_size):
	height=pic.shape[0]
	width=pic.shape[1]
	h=height%patch_size   
	w=width%patch_size
	cond=0;cond1=0
	#patch=queue.Queue(maxsize=patch_size*patch_size)
	temp=np.zeros(())

	i=0;j=0;
	f=open(file,"a")                                                  #open the file where mean and variance pair are to be stacked
	while i<height-h:
		j=0
		while j< width-w:
			r=i;mean1=0.000;sum1=0;sq_sum1=0;var1=0.000;k=0;
			while r<i+patch_size:
				c=j
				while c<j+patch_size:
					sum1 += int(pic[r][c])                             #sum of pixels in 7*7 patch
					#print(int(pic[r][c]))
					sq_sum1 += (int(pic[r][c])*int(pic[r][c]))         #square sum of the pixels in 7*7 patch        
					#ans=(int(pic[r][c])*int(pic[r][c]))
					#print(pic[r][c],ans)
					k+=1
					c+=1
				r+=1

			mean1=(sum1*1.00000)/(patch_size*patch_size)              # calulating maen of pixels values in each 7*7 overlapping patch
			#print(sq_sum1,mean1)
			#var1=0;
			#print(sum1,sq_sum1)
			var1=((sq_sum1*1.00000)/(patch_size*patch_size))-(mean1*mean1)  # calulating variance of pixels values in each 7*7 overlapping patch
			#print(mean1,var1)
			f.write("%lf "%mean1)
			f.write("%lf"%var1)
			f.write("\n")
			j+=patch_size
			if j>width-w-1 and cond1==0 and width%patch_size!=0:                               #this loop ensures that no pixels is left even if there is not 32*32 patch we make we take some pixels from prev patch make 32*32 patch                           
				j=width-patch_size
				cond1=1	
		i+=patch_size
		cond1=0
		if i>height-h-1 and cond==0 and height%patch_size!=0:
			i=height-patch_size
			cond=1
			cond1=0	


	f.close()	


#source path and folder of the cell images
source_path=raw_input("enter the path to folder where trainig images are present:")
#output path and folder where the output to be stacked
destination=raw_input("enter the path to the file where feature vectors are to stored")
#patch size
patch_size=7

extract_feature(pic,destination,patch_size)

i=1
for img in glob.glob(source_path+"/*.png"):
	pic= imageio.imread(img)
	print(pic.shape[0])
	print(pic.shape[1])
	#print('Shape of the image : {}'.format(pic.shape))
	extract_feature(pic,destination,patch_size)
	print ("image completed",i)
	i+=1



