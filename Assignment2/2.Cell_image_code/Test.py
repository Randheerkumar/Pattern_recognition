'''
Pattern recognition cs669

lab 2

Group10

'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt  
import imageio                        # for find_covting
import pickle
import find_cov
import k_means
import gmm_cluster
import test

file=raw_input("enter the image name for which segmentation has to be done")
pic= imageio.imread(file)
patch_size=7;
d=2

with open('k_means_mean','rb') as f1:
	mean=pickle.load(f1)

test.test_img(pic,file,7,mean)


with open('gmm_mean','rb') as f2:
	mean=pickle.load(f2)

with open('gmm_cov_mat','rb') as f3:
	cov_mat=pickle.load(f3)

with open('gmm_pi_k','rb') as f4:
	pi_k=pickle.load(f4)	


#find_cov.find_cov_matrix(x,n,k,d,znk,Pi_k,mean,cov_mat)
#test.test_img(pic,file,7,mean)

test.test_img_gmm(pi_k,mean,cov_mat,pic,patch_size,d)
'''
find_cov.cov_mat_new(x,n,k,d,znk,Pi_k,mean,cov_mat)


gmm_cluster.Gmm(x,n,d,k,znk,Pi_k,mean,cov_mat)
test.test_img(pic,file,7,mean)
'''

