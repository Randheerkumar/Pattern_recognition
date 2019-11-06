#importing the required librabries
import imageio
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from PIL import Image


'''function 'extract_feature' take two parameters pic and file .pic is the three dimensional pexels value of RGB colour of coloured image and
file the a text file where the output histogram represtion into 24 dimesional vector is written'''

def extract_feature(pic,file,patch_size):
	height=pic.shape[0]                                               #no of row of pixels values
	width=pic.shape[1]                                                #no of column of pixels values
	h=height%patch_size   
	w=width%patch_size

	a=np.zeros(24)                                                    # arra to contain 24 dimensional histogram feature vector
	cond=0;cond1=0
	i=0;j=0;

	f=open(file,"w+")                                                 #open the file where the 24 histogram  feature vector is stored
	while i<height-h:
		j=0
		while j<width-w:
			r=i
			while r<i+32:
				c=j
				while c<j+32:                                         # through this loop of we are extracting a feature vector 
					a[pic[r][c][0]/32]+=1
					a[8+(pic[r][c][1]/32)]+=1
					a[16+(pic[r][c][2]/32)]+=1
					c+=1
				r+=1

			for r in range(24):                                        #writing the each componets of a feature vector in a file
				f.write("%d "%a[r])
				a[r]=0
			f.write("\n")	
			j=j+32
			if j>width-w-1 and cond1==0:                               #this loop ensures that no pixels is left even if there is not 32*32 patch we make we take some pixels from prev patch make 32*32 patch                           
				j=width-32
				cond1=1	
		i=i+32
		cond1=0
		if i>height-h-1 and cond==0:
			i=height-32
			cond=1
			cond1=0


	f.close()	


#taking the path and folder where  the images(from which we want to extract feature vector)  are present
source_path="train/rock_arch"

#taking the path and folder where the output feature vectors are to be written
destination_path="train/rock_archF/"
#patch size from which we need to make a 24 dimensional feature vector
patch_size=32
i=1;
for img in glob.glob(source_path+"/*.jpg"):   # reading each images from the source folder
	pic= imageio.imread(img)
	st="img"+str(i)+".txt"
	print(pic.shape[0])
	print(pic.shape[1])
	destination=destination_path+st           #destination file where i need to output the feature vector 
	extract_feature(pic,destination,patch_size)
	print("image completed",i)
	i=i+1
	





'''	print('Type of the image : ' , type(pic))
	print()
	print('Shape of the image : {}'.format(pic.shape))
	print('Image Hight {}'.format(pic.shape[0]))
	print('Image Width {}'.format(pic.shape[1]))
	print('Dimension of Image {}'.format(pic.ndim))
'''